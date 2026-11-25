import io
import json
import sys
import tempfile
import unittest
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'study'))
from server import Handler


class ToolProcess:
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1


class StudyBridgeTests(unittest.TestCase):
    def setUp(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        self.server = SimpleNamespace(output=Path(output.name), server_port=8765,
                                      tool_sessions={}, tool_sequences=count(1))
        self.identity = {'study_id': 'P1', 'session_id': 'survey1'}

    def request(self, path, payload=None, *, body=None, origin=None):
        handler = Handler.__new__(Handler)
        handler.server = self.server
        handler.path = path
        result = []
        handler.reply = lambda code, data: result.append((code, data))
        if payload is None and body is None:
            handler.do_GET()
        else:
            if body is None:
                body = json.dumps(payload).encode()
            handler.headers = {'Content-Length': str(len(body)), 'Host': '127.0.0.1:8765'}
            if origin:
                handler.headers['Origin'] = origin
            handler.rfile = io.BytesIO(body)
            handler.do_POST()
        return result[0]

    def handoff(self, scenario, prepare=False, **identity):
        return self.request('/api/handoff', self.identity | identity | dict(scenario=scenario, prepare=prepare, page=3))

    def tool(self, scenario, sequence, state, acknowledged=0, **identity):
        return self.request('/api/tool', self.identity | identity |
                            dict(scenario=scenario, sequence=sequence, state=state, acknowledged=acknowledged))[1]

    def status(self, scenario, session='survey1'):
        return self.request(f'/api/tool?session_id={session}&scenario={scenario}')[1]

    @patch('server.subprocess.Popen', side_effect=lambda *args, **kwargs: ToolProcess())
    def test_two_scenarios_share_recording_responses_and_identity(self, popen):
        self.assertEqual(self.request('/api/recordings?session=survey1&study_id=P1&recording=rec1&part=0&mime=video/webm',
                                      body=b'recording')[0], 200)
        self.assertEqual(self.handoff('B', prepare=True), (200, {'sequence': 1, 'state': 'preparing'}))
        self.assertEqual(self.tool('B', 1, 'ready'), {'command': None, 'accepted': False})
        self.assertEqual(self.status('B')['state'], 'ready')
        self.assertEqual(self.handoff('B'), (200, {'sequence': 1, 'state': 'requested'}))
        self.assertEqual(self.handoff('B')[1]['sequence'], 1)
        command = self.tool('B', 1, 'ready')['command']
        self.assertEqual((command['study_id'], command['session_id'], command['scenario']), ('P1', 'survey1', 'B'))
        self.tool('B', 1, 'active', acknowledged=1)
        self.assertEqual(self.handoff('A', prepare=True)[0], 409)
        self.assertFalse(self.tool('B', 99, 'complete', acknowledged=99)['accepted'])
        self.assertTrue(self.tool('B', 1, 'complete', acknowledged=1)['accepted'])
        self.assertEqual(self.status('B')['state'], 'finishing')
        self.server.tool_sessions['survey1', 'B']['process'].returncode = 0
        self.assertEqual(self.status('B')['state'], 'complete')
        self.assertEqual(self.handoff('A', prepare=True, study_id='different')[0], 400)
        self.assertEqual(self.handoff('A', prepare=True), (200, {'sequence': 2, 'state': 'preparing'}))
        self.assertEqual(self.handoff('B', prepare=True)[1]['state'], 'complete')
        self.assertEqual(self.status('A')['state'], 'preparing')
        self.assertEqual(self.request('/api/responses', self.identity | {'answers': {}})[0], 200)

        folder = next(self.server.output.glob('P1-*'))
        self.assertEqual(json.loads((folder / 'session.json').read_text()), self.identity)
        self.assertEqual(json.loads((folder / 'rec1.json').read_text())['session_id'], 'survey1')
        self.assertEqual(json.loads((folder / f'{folder.name}.json').read_text())['study_id'], 'P1')
        for call, scenario, attempt in zip(popen.call_args_list, ('B_Cifar', 'A_Pamap'), (1, 2)):
            arguments = call.args[0]
            destination = folder / 'tool' / scenario / f'attempt-{attempt}'
            self.assertIn(f'--study-output={destination}', arguments)
            self.assertIn('--survey-session=survey1', arguments)
            self.assertIn('--id=P1', arguments)
            self.assertTrue((destination / 'process.log').is_file())

    @patch('server.subprocess.Popen', side_effect=lambda *args, **kwargs: ToolProcess())
    def test_retry_cancellation_and_reused_study_id(self, popen):
        self.assertEqual(self.handoff('A', prepare=True)[1]['sequence'], 1)
        first = self.server.tool_sessions['survey1', 'A']['process']
        self.assertEqual(self.handoff('B', prepare=True)[1]['sequence'], 2)
        self.assertIsNotNone(first.poll())
        self.server.tool_sessions['survey1', 'B']['process'].returncode = 1
        self.assertEqual(self.status('B')['state'], 'error')
        self.assertEqual(self.handoff('B')[1]['sequence'], 3)
        self.assertEqual(self.tool('B', 2, 'complete', acknowledged=2), {'command': None, 'accepted': False})
        self.assertEqual(self.handoff('B', prepare=True, session_id='survey2')[1]['sequence'], 4)
        self.assertEqual(self.request('/api/tool/cancel', self.identity)[0], 200)
        self.assertEqual(self.status('B')['state'], 'error')
        self.assertEqual(self.status('B', 'survey2')['state'], 'preparing')
        self.assertEqual(len(list(self.server.output.glob('P1-*'))), 2)

    def test_external_origin_cannot_request_tool_focus(self):
        self.assertEqual(self.request('/api/handoff', self.identity, origin='https://example.com')[0], 403)


if __name__ == '__main__':
    unittest.main()
