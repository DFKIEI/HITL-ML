"""Shared visual theme for the Tkinter UI: a light, high-contrast palette,
larger accessible fonts, and consistent ttk styling, applied once via
apply_theme(root). Centralizing this here - instead of scattering styling
knobs across each window module - is what keeps every window (the main
control panel, the LLM Suggestions window, the initial setup dialog)
visually consistent.

Colors are chosen for WCAG AA contrast (>= 4.5:1 for normal text): TEXT on
BG/SURFACE is ~16:1, TEXT_MUTED on BG is ~5.8:1, and white button text on
PRIMARY/DANGER/SUCCESS is >= 5:1.
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------- palette
BG = "#F4F6F9"
SURFACE = "#FFFFFF"
BORDER = "#D6DBE3"
BORDER_STRONG = "#B7BFCC"

TEXT = "#182233"
TEXT_MUTED = "#57616F"
TEXT_ON_ACCENT = "#FFFFFF"

PRIMARY = "#2456C9"
PRIMARY_HOVER = "#1B419E"
PRIMARY_ACTIVE = "#153581"
PRIMARY_SOFT = "#E4EBFC"  # light tint for selected/hover backgrounds

SUCCESS = "#146B3A"
SUCCESS_SOFT = "#E1F3E8"
DANGER = "#C33B2E"
DANGER_SOFT = "#FBE9E7"
WARNING = "#8A5A00"
WARNING_SOFT = "#FBF0DC"

DISABLED_BG = "#E7EAEF"
DISABLED_FG = "#98A1AD"

# ---------------------------------------------------------------- fonts
# Tk silently substitutes a system default when a family isn't installed, but
# we still probe tkfont.families() so the *first* real match wins instead of
# whatever Tk happens to fall back to.
_FONT_CANDIDATES = ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
_family = None


def _pick_family():
    try:
        available = set(tkfont.families())
    except tk.TclError:
        return "TkDefaultFont"
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return "TkDefaultFont"


def fonts():
    """(body, body_bold, heading, small, ...) font tuples. Call only after a
    Tk root exists - tkfont.families() needs one."""
    global _family
    if _family is None:
        _family = _pick_family()
    return {
        'body': (_family, 11),
        'body_bold': (_family, 11, 'bold'),
        'heading': (_family, 13, 'bold'),
        'title': (_family, 16, 'bold'),
        'small': (_family, 10),
        'small_italic': (_family, 10, 'italic'),
    }


PAD_S = 4
PAD_M = 8
PAD_L = 14


def apply_theme(root):
    """Apply the palette/fonts/ttk styling to `root`. ttk styles are shared
    per interpreter (not per window), so this only needs to run once, before
    the first Toplevel (e.g. the LLM Suggestions window) is built."""
    f = fonts()
    root.configure(bg=BG)

    # Classic (non-ttk) widgets - tk.Scale, tk.Canvas, tk.Checkbutton, the
    # scrolledtext log, the raw tk.Label/Entry/Button in the initial setup
    # dialog - pick these defaults up automatically via the option database.
    root.option_add('*Font', f['body'])
    root.option_add('*Background', BG)
    root.option_add('*Foreground', TEXT)
    root.option_add('*selectBackground', PRIMARY_SOFT)
    root.option_add('*selectForeground', TEXT)
    root.option_add('*Entry.Background', SURFACE)
    root.option_add('*Entry.relief', 'solid')
    root.option_add('*Entry.borderWidth', 1)
    root.option_add('*Entry.highlightThickness', 1)
    root.option_add('*Entry.highlightBackground', BORDER)
    root.option_add('*Entry.highlightColor', PRIMARY)
    root.option_add('*Button.relief', 'flat')
    root.option_add('*Button.borderWidth', 0)
    root.option_add('*Button.padY', 6)
    root.option_add('*Button.padX', 12)
    root.option_add('*Button.background', SURFACE)
    root.option_add('*Button.activeBackground', PRIMARY_SOFT)
    root.option_add('*Button.highlightThickness', 0)
    root.option_add('*Checkbutton.background', BG)
    root.option_add('*Checkbutton.activeBackground', BG)
    root.option_add('*Canvas.background', BG)
    root.option_add('*Canvas.highlightThickness', 0)
    root.option_add('*Listbox.background', SURFACE)
    root.option_add('*Listbox.selectBackground', PRIMARY)
    root.option_add('*Listbox.selectForeground', TEXT_ON_ACCENT)
    root.option_add('*Toplevel.background', BG)

    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    style.configure('.', background=BG, foreground=TEXT, font=f['body'])

    style.configure('TFrame', background=BG)
    style.configure('Surface.TFrame', background=SURFACE)

    style.configure('TLabel', background=BG, foreground=TEXT, font=f['body'])
    style.configure('Muted.TLabel', background=BG, foreground=TEXT_MUTED, font=f['small_italic'])
    style.configure('Heading.TLabel', background=BG, foreground=TEXT, font=f['heading'])
    style.configure('Title.TLabel', background=BG, foreground=TEXT, font=f['title'])
    style.configure('Status.TLabel', background=BG, foreground=TEXT_MUTED, font=f['small'])
    style.configure('Success.TLabel', background=BG, foreground=SUCCESS, font=f['body'])
    style.configure('Danger.TLabel', background=BG, foreground=DANGER, font=f['body'])

    style.configure('TLabelframe', background=BG, bordercolor=BORDER,
                    relief='solid', borderwidth=1)
    style.configure('TLabelframe.Label', background=BG, foreground=PRIMARY, font=f['body_bold'])

    style.configure('TButton', background=SURFACE, foreground=TEXT, font=f['body'],
                    padding=(14, 8), relief='flat', bordercolor=BORDER, borderwidth=1)
    style.map('TButton',
             background=[('disabled', DISABLED_BG), ('pressed', PRIMARY_SOFT), ('active', PRIMARY_SOFT)],
             foreground=[('disabled', DISABLED_FG)],
             bordercolor=[('focus', PRIMARY), ('!focus', BORDER)])

    style.configure('Primary.TButton', background=PRIMARY, foreground=TEXT_ON_ACCENT,
                    font=f['body_bold'], padding=(14, 9), relief='flat', borderwidth=0)
    style.map('Primary.TButton',
             background=[('disabled', DISABLED_BG), ('pressed', PRIMARY_ACTIVE), ('active', PRIMARY_HOVER)],
             foreground=[('disabled', DISABLED_FG)])

    style.configure('Danger.TButton', background=DANGER, foreground=TEXT_ON_ACCENT,
                    font=f['body_bold'], padding=(14, 9), relief='flat', borderwidth=0)
    style.map('Danger.TButton',
             background=[('disabled', DISABLED_BG), ('pressed', '#8E2A20'), ('active', '#A93226')],
             foreground=[('disabled', DISABLED_FG)])

    style.configure('TEntry', fieldbackground=SURFACE, foreground=TEXT, bordercolor=BORDER,
                    lightcolor=SURFACE, darkcolor=BORDER, borderwidth=1, padding=6, font=f['body'])
    style.map('TEntry',
             bordercolor=[('focus', PRIMARY), ('disabled', BORDER)],
             fieldbackground=[('disabled', DISABLED_BG)])

    style.configure('TCombobox', fieldbackground=SURFACE, background=SURFACE, foreground=TEXT,
                    bordercolor=BORDER, arrowcolor=PRIMARY, padding=6, font=f['body'])
    style.map('TCombobox',
             bordercolor=[('focus', PRIMARY)],
             fieldbackground=[('readonly', SURFACE), ('disabled', DISABLED_BG)],
             foreground=[('disabled', DISABLED_FG)])
    root.option_add('*TCombobox*Listbox.background', SURFACE)
    root.option_add('*TCombobox*Listbox.selectBackground', PRIMARY)
    root.option_add('*TCombobox*Listbox.selectForeground', TEXT_ON_ACCENT)
    root.option_add('*TCombobox*Listbox.font', f['body'])

    style.configure('TNotebook', background=BG, bordercolor=BORDER, tabmargins=(2, 4, 2, 0))
    style.configure('TNotebook.Tab', background=SURFACE, foreground=TEXT_MUTED,
                    font=f['body_bold'], padding=(18, 9), bordercolor=BORDER)
    style.map('TNotebook.Tab',
             background=[('selected', PRIMARY_SOFT)],
             foreground=[('selected', PRIMARY)])

    style.configure('TSeparator', background=BORDER)

    style.configure('TScrollbar', background=SURFACE, troughcolor=BG, bordercolor=BG,
                    arrowcolor=TEXT_MUTED, relief='flat')
    style.map('TScrollbar', background=[('active', BORDER_STRONG)])

    style.configure('Horizontal.TScale', background=BG, troughcolor=BORDER)
    style.configure('TCheckbutton', background=BG, foreground=TEXT, font=f['body'])
    style.map('TCheckbutton', background=[('active', BG)])

    style.configure('TProgressbar', background=PRIMARY, troughcolor=BORDER)

    return style


def style_tk_scale(scale_widget):
    """Recolor a classic tk.Scale - ttk.Scale can't show the live numeric
    label these sliders rely on, so we keep tk.Scale but reskin it."""
    f = fonts()
    scale_widget.configure(
        bg=SURFACE, fg=TEXT, troughcolor=BORDER, activebackground=PRIMARY,
        highlightthickness=0, bd=0, font=f['small'], sliderrelief='flat',
        showvalue=True,
    )


def style_scrolled_text(widget):
    """Recolor a scrolledtext.ScrolledText / tk.Text log area."""
    f = fonts()
    widget.configure(
        bg=SURFACE, fg=TEXT, insertbackground=TEXT, relief='solid',
        borderwidth=1, highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=PRIMARY, font=f['small'], padx=8, pady=6,
    )
