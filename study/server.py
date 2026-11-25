"""Local survey and recording storage. Run: python study/server.py"""
import argparse,json,os,re,subprocess,sys,threading
from itertools import count
from datetime import datetime,timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
STUDY_SCENARIOS={'A':('A_Pamap','PAMAP2'),'B':('B_Cifar','CIFAR10')}
def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise ValueError('Invalid session identifier')
    return value
def study_label(value):
    return re.sub(r'[^A-Za-z0-9_-]+','-',str(value)).strip('-_')[:80] or 'participant'

def validate_study_id(value):
    key=str(value).strip()
    if not key or len(key)>80 or re.search(r'[\\/:*?"<>|\x00-\x1f]',key):
        raise ValueError('Enter a Study ID of up to 80 characters without path punctuation.')
    return key

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
    study_id=validate_study_id(study_id)
    legacy=output/sid
    if legacy.is_dir():
        folder=legacy
    else:
        index=output/'.sessions'
        index.mkdir(exist_ok=True)
        entry=index/sid
        if entry.is_file():
            folder=output/entry.read_text()
        else:
            folder=available_path(output/f'{study_label(study_id)}-{timestamp()}')
            folder.mkdir()
            temp=entry.with_suffix('.tmp')
            temp.write_text(folder.name)
            temp.replace(entry)
    manifest=folder/'session.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))['study_id']!=study_id:
            raise ValueError('This survey session already has a different Study ID.')
    else:
        manifest.write_text(json.dumps({'study_id':study_id,'session_id':sid},indent=2),encoding='utf-8')
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
            query=parse_qs(parsed.query)
            with LOCK:
                entry=self.server.tool_sessions.get((query.get('session_id',[''])[0],query.get('scenario',[''])[0]))
                status=self.tool_status(entry)
            return self.reply(200,status)
        return super().do_GET()

    def tool_status(self,entry):
        if entry is None:
            return {}
        status=entry['status']
        if status['state'] not in ('complete','error') and entry['process'].poll() is not None:
            status.update(state='error',reason='The study tool closed before completing the task. Click Open tool again to retry.')
        if status['state']=='complete' and entry['process'].poll() is None:
            return dict(status,state='finishing')
        return dict(status)

    def launch_tool(self,study_id,sid,scenario,sequence,session_directory):
        scenario_name,dataset=STUDY_SCENARIOS[scenario]
        project=ROOT.parent
        checkpoint=project/'saved_checkpoints'/dataset/'checkpoint_epoch_25.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Missing epoch-25 checkpoint for {dataset}.')
        folder=session_directory/'tool'/scenario_name
        folder.mkdir(parents=True,exist_ok=True)
        attempt=available_path(folder/f'attempt-{sequence}')
        attempt.mkdir()
        environment=dict(os.environ,HITL_SURVEY_URL=f'http://127.0.0.1:{self.server.server_port}')
        with (attempt/'process.log').open('ab') as log:
            return subprocess.Popen(
                [sys.executable,'-u',str(project/'code'/'main.py'),f'--id={study_id}',
                 f'--scenario={scenario_name}',f'--dataset={dataset}',f'--checkpoint={checkpoint}',
                 f'--survey-sequence={sequence}',f'--survey-session={sid}',f'--survey-scenario={scenario}',
                 f'--study-output={attempt}'],
                cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,
            )

    def request_tool(self,payload,sid,study_id,scenario):
        tasks=self.server.tool_sessions
        entry=tasks.get((sid,scenario))
        prepare=payload.get('prepare') is True
        session_directory=session_folder(self.server.output,sid,study_id)
        if entry is not None and entry['status']['state']=='complete':
            return 200,self.tool_status(entry)
        for (other_sid,other_scenario),other in tasks.items():
            if other_sid!=sid or other_scenario==scenario:
                continue
            state=self.tool_status(other)['state']
            if state in ('requested','active','finishing'):
                return 409,{'error':'Finish the current tool task before opening the next one.'}
            if state in ('preparing','ready'):
                other['process'].terminate()
                other['status'].update(state='error',reason='The scenario order changed.')
        if entry is None or self.tool_status(entry)['state']=='error':
            sequence=next(self.server.tool_sequences)
            process=self.launch_tool(study_id,sid,scenario,sequence,session_directory)
            entry=tasks[sid,scenario]={
                'study_id':study_id,'process':process,'command':None,
                'status':{'sequence':sequence,'state':'preparing'},
            }
        status=entry['status']
        if not prepare and entry['command'] is None:
            entry['command']={'sequence':status['sequence'],'action':'focus_tool',
                              'page':payload.get('page'),'session_id':sid,
                              'study_id':study_id,'scenario':scenario}
            status['state']='requested'
        return 200,{'sequence':status['sequence'],'state':status['state']}

    def tool_update(self,payload,entry):
        if entry is None or payload.get('sequence')!=entry['status']['sequence']:
            return {'command':None,'accepted':False}
        status=entry['status']
        acknowledged=payload.get('acknowledged')==status['sequence']
        reported=payload.get('state')
        if status['state']=='preparing' and reported=='ready':
            status['state']='ready'
        if acknowledged and entry['command'] and status['state'] in ('requested','active') and reported in ('active','complete'):
            status['state']=reported
            for field in ('interactions','epoch','reason'):
                if field in payload:
                    status[field]=payload[field]
        return {'command':None if acknowledged else entry['command'],
                'accepted':acknowledged and status['state']=='complete'}

    def do_POST(self):
        try:
            parsed=urlparse(self.path);length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>32*1024*1024:raise ValueError('Request size invalid')
            body=self.rfile.read(length)
            if parsed.path in ('/api/tool','/api/handoff','/api/tool/cancel'):
                origin=self.headers.get('Origin')
                if origin and origin != 'http://'+self.headers.get('Host',''):
                    return self.reply(403,{'error':'Local survey origin required'})
                payload=json.loads(body)
                sid=identifier(payload['session_id'])
                with LOCK:
                    if parsed.path=='/api/tool/cancel':
                        for (session_id,_),entry in self.server.tool_sessions.items():
                            if session_id==sid and entry['status']['state']!='complete' and entry['process'].poll() is None:
                                entry['process'].terminate()
                                entry['status'].update(state='error',reason='The survey was closed.')
                        return self.reply(200,{'cancelled':True})
                    study_id=validate_study_id(payload.get('study_id',''))
                    scenario=payload.get('scenario')
                    if scenario not in STUDY_SCENARIOS:
                        raise ValueError('Scenario must be A or B')
                    entry=self.server.tool_sessions.get((sid,scenario))
                    if entry is not None and entry['study_id']!=study_id:
                        raise ValueError('This survey session already has a different Study ID.')
                    if parsed.path=='/api/handoff':
                        code,result=self.request_tool(payload,sid,study_id,scenario)
                    else:
                        code,result=200,self.tool_update(payload,entry)
                return self.reply(code,result)
            if parsed.path=='/api/responses':
                payload=json.loads(body);sid=identifier(payload['session_id'])
                payload['study_id']=validate_study_id(payload.get('study_id',''))
                if not isinstance(payload.get('answers'),dict):raise ValueError('Answers missing')
                payload['saved_at_utc']=datetime.now(timezone.utc).isoformat()
                with LOCK:
                    folder=session_folder(self.server.output,sid,payload.get('study_id',''))
                    response=folder/('response.json' if folder.name==sid else f'{folder.name}.json')
                    temp=folder/'response.tmp';temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(response)
                return self.reply(200,{'saved':True,'session_id':sid})
            if parsed.path=='/api/recordings':
                q=parse_qs(parsed.query);sid=identifier(q['session'][0]);rid=identifier(q['recording'][0]);part=int(q['part'][0])
                study_id=validate_study_id(q.get('study_id',[''])[0])
                mime=q['mime'][0].split(';')[0]
                if mime not in ('video/webm','video/mp4') or part<0:raise ValueError('Invalid media metadata')
                with LOCK:
                    folder=session_folder(self.server.output,sid,study_id)
                    manifest=folder/(rid+'.json')
                    meta=json.loads(manifest.read_text()) if manifest.exists() else {'parts':0,'mime':mime,'created_at_utc':datetime.now(timezone.utc).isoformat()}
                    meta.update(study_id=study_id,session_id=sid)
                    if part!=meta['parts']:raise ValueError('Recording part out of order')
                    extension='webm' if mime=='video/webm' else 'mp4'
                    if not manifest.exists():
                        video=available_path(folder/f'{study_label(study_id)}-{timestamp()}.{extension}')
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
    parser.add_argument('--output',type=Path,default=ROOT.parent/'full_study_results');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.output=args.output.resolve()
    server.tool_sessions={};server.tool_sequences=count(1)
    print(f'Local survey: http://127.0.0.1:{args.port}\nResponses and recordings: {server.output}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            for entry in server.tool_sessions.values():
                if entry['process'].poll() is None:
                    entry['process'].terminate()
        server.server_close()
