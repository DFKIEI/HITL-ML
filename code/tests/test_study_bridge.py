import io
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'study'))
from server import Handler,TOOL_SESSIONS


def post(path,payload,origin=None):
    handler=Handler.__new__(Handler)
    body=json.dumps(payload).encode()
    handler.path=path
    handler.headers={'Content-Length':str(len(body)),'Host':'127.0.0.1:8765'}
    if origin:handler.headers['Origin']=origin
    handler.rfile=io.BytesIO(body)
    result=[]
    handler.reply=lambda code,data:result.append((code,data))
    handler.do_POST()
    return result[0]


def test_handoff_acknowledgement_and_study_isolation():
    TOOL_SESSIONS.clear()
    try:
        assert post('/api/handoff',dict(study_id='P1',session_id='survey1',page=3))[1]['sequence']==1
        response=post('/api/tool',dict(study_id='P1',acknowledged=0,state='ready'))[1]
        assert response['command']['session_id']=='survey1'
        assert post('/api/tool',dict(study_id='P2',acknowledged=0))[1]['command'] is None
        assert post('/api/tool',dict(study_id='P1',acknowledged=1,state='complete',interactions=3))[1]['command'] is None
        assert TOOL_SESSIONS['P1']['state']=='complete'
        assert post('/api/handoff',dict(study_id='P1',page=4))[1]['sequence']==2
        stale=post('/api/tool',dict(study_id='P1',acknowledged=1,state='complete'))[1]
        assert stale['command']['sequence']==2
        assert TOOL_SESSIONS['P1']['state']=='requested'
    finally:TOOL_SESSIONS.clear()


def test_external_origin_cannot_request_tool_focus():
    assert post('/api/handoff',dict(study_id='P1'),origin='https://example.com')[0]==403
