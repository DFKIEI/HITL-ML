"""Local survey/tool handoff. Network I/O never runs on Tk's event thread."""
import json
import os
import queue
import threading
from urllib.request import Request, urlopen


class StudyBridge:
    def __init__(self, ui):
        self.ui = ui
        self.results = queue.Queue()
        self.busy = False
        self.sequence = 0
        self.managed_sequence = None
        self.scenario = None
        self.completion_sent = threading.Event()
        self.status = {'state': 'ready'}
        self.url = os.environ.get('HITL_SURVEY_URL', 'http://127.0.0.1:8765').rstrip('/')
        self.focused = None
        ui.root.bind('<FocusIn>', lambda e: ui.root.after_idle(self.log_focus), add='+')
        ui.root.bind('<FocusOut>', lambda e: ui.root.after_idle(self.log_focus), add='+')
        ui.root.after(250 if ui.study_session_id else 1000, self.poll)

    def log_focus(self):
        if self.ui.events.closed:
            return
        focused = self.ui.root.focus_displayof() is not None
        if focused != self.focused:
            self.focused = focused
            self.ui.events.log('TOOL_FOCUS_CHANGED', focused=focused)

    def publish(self, state, **details):
        self.status = dict(state=state, **details)

    def poll(self):
        ui = self.ui
        if self.completion_sent.is_set():
            ui.on_close()
            return
        try:
            command = self.results.get_nowait()
        except queue.Empty:
            command = None
        if (command and command.get('sequence', 0) > self.sequence
                and self.managed_sequence in (None, command['sequence'])):
            self.sequence = command['sequence']
            if not ui.study_mode_var.get():
                ui.study_mode_var.set(True)
                ui.toggle_study_mode()
            ui.events.log('SURVEY_TOOL_HANDOFF', **command)
            ui.begin_study_task()
            ui.root.deiconify()
            ui.maximize_window()
            ui.root.lift()
            ui.root.attributes('-topmost', True)
            ui.root.after(250, lambda: ui.root.attributes('-topmost', False))
            ui.root.focus_force()
        if self.managed_sequence is not None and not self.busy:
            self.busy = True
            payload = dict(study_id=str(ui.probant_id), session_id=ui.study_session_id,
                           scenario=self.scenario, sequence=self.managed_sequence,
                           acknowledged=self.sequence, **self.status)
            def worker():
                try:
                    request = Request(self.url+'/api/tool', data=json.dumps(payload).encode(),
                                      headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=1) as response:
                        result = json.load(response)
                    if payload['state'] == 'complete' and result.get('accepted'):
                        self.completion_sent.set()
                    if result.get('command'):
                        self.results.put(result['command'])
                except (OSError, ValueError):
                    pass  # The desktop application also works without a survey server.
                finally:
                    self.busy = False
            threading.Thread(target=worker, daemon=True).start()
        ui.root.after(250 if self.managed_sequence is not None else 1000, self.poll)
