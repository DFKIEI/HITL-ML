"""Tk-only checkpoint browser: avoids the macOS native open-panel stall."""
import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk

PROJECT = Path(__file__).resolve().parents[2]
EXTENSIONS = {'.pt', '.pth', '.ckpt'}


def resolve_selection(text):
    """Validate a pasted path without loading a model or opening a native dialog."""
    path = Path(os.path.expandvars(text.strip().strip('"').strip("'"))).expanduser()
    if not path.exists():
        raise ValueError('That path does not exist.')
    if not path.is_file():
        raise ValueError('Choose a checkpoint file, not a folder.')
    return path.resolve()


def list_directory(folder):
    folder = Path(folder)
    entries = [p for p in folder.iterdir() if p.is_dir() or p.suffix.lower() in EXTENSIONS]
    def natural(name):
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]
    return sorted(entries, key=lambda p: (not p.is_dir(), natural(p.name)))


class CheckpointPicker:
    def __init__(self, parent, initialdir=None, title='Choose pretrained checkpoint'):
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry('760x500')
        self.window.minsize(520, 350)
        self.window.transient(parent)
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)
        self.folder = Path(initialdir or PROJECT / 'saved_checkpoints').expanduser().resolve()
        if not self.folder.is_dir():
            self.folder = PROJECT
        self.path = tk.StringVar(value=str(self.folder))
        self.status = tk.StringVar(value='Double-click a folder to open it, or choose a checkpoint file.')
        content = ttk.Frame(self.window, padding=16)
        content.pack(fill=tk.BOTH, expand=True)
        ttk.Label(content, text='Checkpoint file or folder path').pack(anchor=tk.W)
        row = ttk.Frame(content)
        row.pack(fill=tk.X, pady=(4, 10))
        self.entry = ttk.Entry(row, textvariable=self.path)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind('<Return>', lambda _event: self.open_path())
        ttk.Button(row, text='Open path', command=self.open_path).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text='Up', command=self.up).pack(side=tk.LEFT, padx=(8, 0))
        area = ttk.Frame(content)
        area.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(area, columns=('kind',), show='tree headings', selectmode='browse')
        self.tree.heading('#0', text='File / folder')
        self.tree.heading('kind', text='Type')
        self.tree.column('#0', width=480)
        self.tree.column('kind', width=100, stretch=False)
        scrollbar = ttk.Scrollbar(area, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind('<<TreeviewSelect>>', self.select)
        self.tree.bind('<Double-1>', lambda _event: self.open_selected())
        self.tree.bind('<Return>', lambda _event: self.open_selected())
        ttk.Label(content, textvariable=self.status, wraplength=690).pack(anchor=tk.W, pady=(10, 4))
        buttons = ttk.Frame(content)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text='Cancel', command=self.cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text='Use checkpoint', command=self.choose).pack(side=tk.RIGHT, padx=8)
        self.window.bind('<Escape>', lambda _event: self.cancel())
        self.refresh()

    def refresh(self):
        try:
            entries = list_directory(self.folder)
        except OSError as exc:
            self.status.set(f'Could not read folder: {exc}')
            return
        self.tree.delete(*self.tree.get_children())
        self.entries = {str(i): p for i, p in enumerate(entries)}
        for key, path in self.entries.items():
            self.tree.insert('', tk.END, iid=key, text=path.name,
                             values=('Folder' if path.is_dir() else 'Checkpoint',))
        self.path.set(str(self.folder))
        self.status.set(f'{self.folder} · {len(entries)} folders/checkpoint files')

    def select(self, _event=None):
        selected = self.tree.selection()
        if selected:
            self.path.set(str(self.entries[selected[0]]))

    def open_selected(self):
        self.select()
        self.open_path()

    def open_path(self):
        path = Path(os.path.expandvars(self.path.get().strip().strip('"').strip("'"))).expanduser()
        if path.is_dir():
            self.folder = path.resolve()
            self.refresh()
        else:
            self.choose()

    def up(self):
        self.folder = self.folder.parent
        self.refresh()

    def choose(self):
        try:
            self.result = str(resolve_selection(self.path.get()))
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))
            return
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()

    def show(self):
        self.window.grab_set()
        self.entry.focus_set()
        self.window.wait_window()
        return self.result


def choose_checkpoint_file(parent, initialdir=None, title='Choose pretrained checkpoint'):
    return CheckpointPicker(parent, initialdir=initialdir, title=title).show()
