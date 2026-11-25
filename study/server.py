"""Local survey and recording storage. Run: python study/server.py"""
import argparse,json,os,re,subprocess,sys,threading
from datetime import datetime,timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from convert_qsf import convert
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
TOOL_SESSIONS={}
STUDY_DATASETS={'A':'PAMAP2','B':'CIFAR10'}
def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise ValueError('Invalid session identifier')
    return value
def study_label(value):
    return re.sub(r'[^A-Za-z0-9_-]+','-',str(value)).strip('-_')[:80] or 'participant'

def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')

def available_path(path):
    candidate=path
    number=2
    while candidate.exists():
        candidate=path.with_name(f'{path.stem}-{number}{path.suffix}')
        number+=1
    return candidate

def session_folder(output,sid,study_id):
    legacy=output/sid
    if legacy.is_dir():
        return legacy
    index=output/'.sessions'
    index.mkdir(exist_ok=True)
    entry=index/sid
    if entry.is_file():
        return output/entry.read_text()
    folder=available_path(output/f'{study_label(study_id)}-{timestamp()}')
    folder.mkdir()
    temp=entry.with_suffix('.tmp')
    temp.write_text(folder.name)
    temp.replace(entry)
    return folder

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'web'),**kwargs)
    def log_message(self,format,*args):pass
    def reply(self,status,payload):
        data=json.dumps(payload).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path=='/api/tool':
            key=parse_qs(parsed.query).get('study_id',[''])[0].strip()
            with LOCK:
                status=dict(TOOL_SESSIONS.get(key,{}))
                launch=getattr(self.server,'tool_launch',None)
                if launch and launch[:2]==(key,status.get('sequence')) and status.get('state')!='complete':
                    if launch[2].poll() is not None:
                        status.update(state='error',reason='The study tool closed before completing the task. Click Open tool again to retry.')
            return self.reply(200,status)
        return super().do_GET()

    def launch_tool(self,key,scenario,sequence):
        if re.search(r'[\\/:*?"<>|\x00-\x1f]',key):
            raise ValueError('Use a Study ID without path punctuation.')
        dataset=STUDY_DATASETS[scenario]
        project=ROOT.parent
        checkpoint=project/'saved_checkpoints'/dataset/'checkpoint_epoch_25.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Missing epoch-25 checkpoint for {dataset}.')
        environment=dict(os.environ,HITL_SURVEY_URL=f'http://127.0.0.1:{self.server.server_port}')
        process=subprocess.Popen(
            [sys.executable,str(project/'code'/'main.py'),f'--id={key}',
             f'--scenario={scenario}-{dataset}',f'--dataset={dataset}',
             f'--checkpoint={checkpoint}',f'--survey-sequence={sequence}'],
            cwd=project,env=environment,
        )
        self.server.tool_launch=(key,sequence,process)

    def do_POST(self):
        try:
            parsed=urlparse(self.path);length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>32*1024*1024:raise ValueError('Request size invalid')
            body=self.rfile.read(length)
            if parsed.path in ('/api/tool','/api/handoff'):
                origin=self.headers.get('Origin')
                if origin and origin != 'http://'+self.headers.get('Host',''):
                    return self.reply(403,{'error':'Local survey origin required'})
                payload=json.loads(body);key=str(payload.get('study_id','')).strip()
                if not key or len(key)>80:raise ValueError('Study ID required')
                with LOCK:
                    entry=TOOL_SESSIONS.setdefault(key,{})
                    if parsed.path=='/api/handoff':
                        scenario=payload.get('scenario')
                        if scenario is not None and scenario not in ('A','B'):
                            raise ValueError('Scenario must be A or B')
                        launch=getattr(self.server,'tool_launch',None)
                        if scenario and launch and launch[2].poll() is None:
                            previous=TOOL_SESSIONS[launch[0]]
                            previous_command=previous.get('command',{})
                            if (launch[0]==key and previous_command.get('session_id')==payload.get('session_id')
                                    and previous_command.get('page')==payload.get('page')
                                    and previous_command.get('scenario')==scenario
                                    and previous.get('state')!='complete'):
                                return self.reply(200,{'requested':True,'sequence':launch[1]})
                            return self.reply(409,{'error':'The previous tool session is still open or closing. Close it, then click Open tool again.'})
                        sequence=entry.get('sequence',0)+1
                        if scenario:
                            self.launch_tool(key,scenario,sequence)
                        entry.update(sequence=sequence, command={'sequence':sequence,'action':'focus_tool',
                                     'page':payload.get('page'),'session_id':payload.get('session_id'),
                                     'study_id':key,'scenario':scenario},
                                     state='requested')
                        result={'requested':True,'sequence':sequence}
                    else:
                        entry['last_seen']=datetime.now(timezone.utc).isoformat()
                        # An old completion must not finish a newly requested task.
                        if payload.get('acknowledged',0)>=entry.get('sequence',0):
                            entry['state']=payload.get('state','ready')
                            for field in ('interactions','epoch','reason'):
                                if field in payload:entry[field]=payload[field]
                        result={'command':entry.get('command') if entry.get('sequence',0)>payload.get('acknowledged',0) else None}
                return self.reply(200,result)
            if parsed.path=='/api/responses':
                payload=json.loads(body);sid=identifier(payload['session_id'])
                if not isinstance(payload.get('answers'),dict):raise ValueError('Answers missing')
                payload['saved_at_utc']=datetime.now(timezone.utc).isoformat()
                with LOCK:
                    folder=session_folder(self.server.output,sid,payload.get('study_id',''))
                    response=folder/('response.json' if folder.name==sid else f'{folder.name}.json')
                    temp=folder/'response.tmp';temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(response)
                return self.reply(200,{'saved':True,'session_id':sid})
            if parsed.path=='/api/recordings':
                q=parse_qs(parsed.query);sid=identifier(q['session'][0]);rid=identifier(q['recording'][0]);part=int(q['part'][0])
                mime=q['mime'][0].split(';')[0]
                if mime not in ('video/webm','video/mp4') or part<0:raise ValueError('Invalid media metadata')
                with LOCK:
                    folder=session_folder(self.server.output,sid,q.get('study_id',[''])[0])
                    manifest=folder/(rid+'.json')
                    meta=json.loads(manifest.read_text()) if manifest.exists() else {'parts':0,'mime':mime,'created_at_utc':datetime.now(timezone.utc).isoformat()}
                    if part!=meta['parts']:raise ValueError('Recording part out of order')
                    extension='webm' if mime=='video/webm' else 'mp4'
                    if not manifest.exists():
                        video=available_path(folder/f'{study_label(q.get("study_id",[""])[0])}-{timestamp()}.{extension}')
                        meta['filename']=video.name
                    filename=meta.get('filename',f'{rid}.{extension}')
                    with (folder/filename).open('ab') as f:f.write(body)
                    meta['parts']+=1;manifest.write_text(json.dumps(meta,indent=2))
                return self.reply(200,{'saved':True,'part':part,'filename':filename})
            self.reply(404,{'error':'Unknown endpoint'})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:self.reply(400,{'error':str(exc)})
        except Exception:self.reply(500,{'error':'Could not save locally. Check server output directory.'})
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--output',type=Path,default=ROOT/'results');args=parser.parse_args()
    convert();args.output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.output=args.output.resolve()
    print(f'Local survey: http://127.0.0.1:{args.port}\nResponses and recordings: {server.output}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
