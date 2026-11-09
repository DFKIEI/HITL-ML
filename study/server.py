"""Local survey and recording storage. Run: python study/server.py"""
import argparse,json,re,threading,uuid,os
from datetime import datetime,timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from convert_qsf import convert
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise ValueError('Invalid session identifier')
    return value
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'web'),**kwargs)
    def log_message(self,format,*args):pass
    def reply(self,status,payload):
        data=json.dumps(payload).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_POST(self):
        try:
            parsed=urlparse(self.path);length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>32*1024*1024:raise ValueError('Request size invalid')
            body=self.rfile.read(length)
            if parsed.path=='/api/responses':
                payload=json.loads(body);sid=identifier(payload['session_id'])
                if not isinstance(payload.get('answers'),dict):raise ValueError('Answers missing')
                payload['saved_at_utc']=datetime.now(timezone.utc).isoformat()
                with LOCK:
                    folder=self.server.output/sid;folder.mkdir(parents=True,exist_ok=True)
                    temp=folder/'response.tmp';temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False));temp.replace(folder/'response.json')
                return self.reply(200,{'saved':True,'session_id':sid})
            if parsed.path=='/api/recordings':
                q=parse_qs(parsed.query);sid=identifier(q['session'][0]);rid=identifier(q['recording'][0]);part=int(q['part'][0])
                mime=q['mime'][0].split(';')[0]
                if mime not in ('video/webm','video/mp4') or part<0:raise ValueError('Invalid media metadata')
                with LOCK:
                    folder=self.server.output/sid;folder.mkdir(parents=True,exist_ok=True)
                    manifest=folder/(rid+'.json')
                    meta=json.loads(manifest.read_text()) if manifest.exists() else {'parts':0,'mime':mime,'created_at_utc':datetime.now(timezone.utc).isoformat()}
                    if part!=meta['parts']:raise ValueError('Recording part out of order')
                    extension='webm' if mime=='video/webm' else 'mp4'
                    with (folder/(rid+'.'+extension)).open('ab') as f:f.write(body)
                    meta['parts']+=1;manifest.write_text(json.dumps(meta,indent=2))
                return self.reply(200,{'saved':True,'part':part})
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
