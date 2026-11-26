"""Tutorial v2: an experiment walkthrough focused on the plot and the right sidebar.
Records the real desktop UI (saved CIFAR10 model), cropped to the centre plot + suggestions
sidebar, with a visible cursor. Shows arrow/circle edits, text editing, applying, and direct
point/cluster moves. The old tutorial (create_tutorial.py, tutorial.mp4) is left untouched.
Runs on macOS (window capture via screencapture) and Windows (screen grab of the app window).
Windows needs ffmpeg on PATH and the existing narration files, so use --reuse-narration (set automatically).
Run from the repository root. Output: study/web/tutorial_v2.{mp4,vtt,m4a}, tutorial_v2_script.json
"""
import os,sys,time,json,subprocess,argparse,math,wave,textwrap
os.environ.setdefault('HITL_NUM_WORKERS','0')  # no loader worker processes: they crash on exit under Windows spawn
if sys.platform=='win32':
    import ctypes
    try:ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixels, so screen grabs match widget geometry
    except Exception:pass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/tools'));os.chdir(ROOT)
import tkinter as tk
import numpy as np
from types import SimpleNamespace
import torch
from PIL import Image,ImageDraw,ImageFont
CACHED_TEXT_REPLY=json.dumps(dict(class_i=3,class_j=5,direction='away_from_j',scale='small',tighten_i=True,tighten_j=False))
from data.data_loader import load_dataset
from model import get_model
from ui.ui import UI
from llm.suggestions import parse_suggestions
from llm.strategies import STRATEGIES
from ui.ui_display import llm_overlay_press,llm_overlay_motion,llm_overlay_release,_llm_handle_at

# (title, spoken text, on-screen caption)
SCENES=[
('About this tool','Welcome. This tool lets you steer how a neural network learns. The plot shows how the model groups its data, with one colour for each class. A language model suggests changes to the plot, such as moving one class away from another. You decide which suggestions to apply, then the model trains with your decisions.','You steer the model: review AI suggestions, then train.',19),
('Reading the screen','Two things help you read the screen. The plot legend explains the symbols: a cross is a class center, a dashed arrow is a previewed move, a dashed circle is the target spread, and an outlined dot is a misclassified point. The training log lists test and validation accuracy after each epoch, so you can see whether your changes help.','Legend: what the symbols mean. Training log: accuracy per epoch.',22),
('1. Get suggestions','Click Get Suggestions. The language model proposes five changes, and each one is previewed on the plot. The Focus box is optional. Use it to tell the model what to concentrate on.','Get Suggestions. The Focus box is optional.',15),
('2. Read a suggestion','Each suggestion covers one pair of classes. The Arrow box moves a class, and the Tighten box makes its cluster more compact. Untick a box to leave that part out. Previous and Next switch between suggestions. Highlight suggested classes focuses the plot on them.','Arrow: move the class. Tighten: shrink its spread. Untick to skip.',20),
('Optional: adjust on the plot','This step is optional. If you want a different move, drag an arrow tip to change its direction and distance, or drag a circle edge to change how tight a class is.','Optional. Arrow tip: direction and distance. Circle edge: tightness.',13),
('Optional: edit the text','Also optional. You can rewrite the suggestion in your own words, then click Update plot from instruction. The arrows and circles update to match.','Optional. Edit the text, then Update plot from instruction.',12),
('3. Accept or reject','Every suggestion needs a decision. Accept applies it to the plot, and Reject discards it. If you are happy with a suggestion as it is, you can accept it straight away, without adjusting anything.','Accept applies a suggestion. Reject discards it.',16),
('Optional: add your own','You can also add your own idea. Click Add my own, choose two classes, describe the change, update the plot, and accept. Click Skip if you have nothing to add.','Optional. Add my own: pick two classes, write the change, accept.',17),
('4. Continue training','When every suggestion is decided, click Continue Training. The model trains with your changes, and the training log shows the new accuracy.','Continue Training starts training with your changes.',11),
]

def ffmpeg_exe():
    import shutil
    found=shutil.which('ffmpeg')
    if found:return found
    try:
        import imageio_ffmpeg  # bundled binary, listed in requirements.txt
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit('ffmpeg not found. Install it (winget install ffmpeg) or pip install imageio-ffmpeg.')

def encode(entries,output,frames,name,frame_count=None):
    manifest=frames/'v2_frames.txt'
    if not manifest.is_file():raise SystemExit(f'{manifest} not found; run the recording first.')
    total=sum(e['duration'] for e in entries);dest=output/f'{name}.mp4'
    subprocess.run([ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-i',str(output/'tutorial_v2_narration.m4a'),
                    '-t',str(total),'-vf','fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p','-c:v','libx264','-preset','fast','-crf','22',
                    '-c:a','aac','-metadata:s:a:0','language=eng','-disposition:a:0','default','-movflags','+faststart',str(dest)],check=True)
    def stamp(sec):return f'{sec//3600:02d}:{sec//60%60:02d}:{sec%60:02d}.000'
    start=0;vtt=['WEBVTT\n']
    for i,e in enumerate(entries):vtt.append(f"{i+1}\n{stamp(start)} --> {stamp(start+e['duration'])}\n{e['title']}: {e['caption']}\n");start+=e['duration']
    (output/f'{name}.vtt').write_text('\n'.join(vtt))
    print(f'Tutorial generated: {dest}; {total} s',flush=True)

def build_narration(frames,output,voice,rate):
    entries=[]
    for i,(title,text,cap,floor) in enumerate(SCENES):
        spoken=text.replace('. ','. [[slnc 250]] ')
        raw=frames/f'v2_narration_{i:02d}.aiff';wav=frames/f'v2_narration_{i:02d}.wav'
        subprocess.run(['say','-v',voice,'-r',str(rate),'-o',str(raw),spoken],check=True)
        subprocess.run(['afconvert','-f','WAVE','-d','LEI16@24000',str(raw),str(wav)],check=True)
        with wave.open(str(wav)) as f:seconds=f.getnframes()/f.getframerate()
        if seconds<1:raise RuntimeError('Speech synthesis returned no usable audio.')
        duration=max(floor,math.ceil(seconds+1.0))
        padded=frames/f'v2_audio_{i:02d}.wav'
        subprocess.run([ffmpeg_exe(),'-y','-v','error','-i',str(wav),'-af','apad','-t',str(duration),str(padded)],check=True)
        entries.append(dict(title=title,caption=text,short=cap,duration=duration,audio=str(padded)))
    manifest=frames/'v2_audio.txt';manifest.write_text(''.join(f"file '{e['audio']}'\n" for e in entries))
    subprocess.run([ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c:a','aac','-b:a','128k',str(output/'tutorial_v2_narration.m4a')],check=True)
    (output/'tutorial_v2_script.json').write_text(json.dumps([{k:v for k,v in e.items() if k!='audio'} for e in entries],indent=2))
    print('Narration ready:',sum(e['duration'] for e in entries),'seconds',flush=True)
    return entries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--narration-only',action='store_true');p.add_argument('--encode-only',action='store_true',help='re-encode the saved frames without recording');p.add_argument('--name',default='tutorial_v2',help='output file name without extension');p.add_argument('--verify-ui',action='store_true')
    p.add_argument('--reuse-narration',action='store_true');p.add_argument('--voice',default='Daniel');p.add_argument('--rate',type=int,default=155)
    args=p.parse_args()
    if sys.platform!='darwin' and not args.verify_ui:args.reuse_narration=True  # speech synthesis uses macOS `say`
    output=ROOT/'study/web';frames=ROOT/'reports/tutorial_frames_v2';frames.mkdir(parents=True,exist_ok=True)
    entries=([dict(title=t,caption=x,short=c,duration=f) for t,x,c,f in SCENES] if args.verify_ui else
             json.loads((output/'tutorial_v2_script.json').read_text()) if args.reuse_narration else
             build_narration(frames,output,args.voice,args.rate))
    if args.narration_only:return
    if args.encode_only:encode(entries,output,frames,args.name);return
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train,val,test,classes,shape=load_dataset('CIFAR10',512)
    model=get_model('CNN_CIFAR10',shape,classes).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    root=tk.Tk();app=UI(root,model,opt,train,val,test,device,'CIFAR10','CNN_CIFAR10','cross_entropy','validation','TUTORIAL','cursor_demo')
    sw,sh=root.winfo_screenwidth(),root.winfo_screenheight();root.geometry(f'{sw-20}x{sh-(110 if sys.platform=="darwin" else 130)}+10+10');root.update()
    app.load_pretrained_model(str(ROOT/'saved_checkpoints/CIFAR10/checkpoint_epoch_25.pt'))
    app.strategy_display_var.set(STRATEGIES[4].label);app.on_strategy_change();app.study_mode_var.set(True);app.toggle_study_mode();root.update()
    names=dict(enumerate(val.dataset.dataset.classes))
    record=next(json.loads(l) for l in (ROOT/'code/experiments/results/llm_comparison.jsonl').read_text().splitlines()
                if json.loads(l)['model']=='openai/gpt-5-mini' and json.loads(l)['dataset']=='CIFAR10')
    summary,suggestions=parse_suggestions(record['raw'],names,set(names))
    app.llm_panel.model_var.set('openai/gpt-5-mini')
    for s in suggestions:s.original_text=s.suggestion
    root.update()
    if not args.verify_ui:
        root.deiconify();root.lift();root.attributes('-topmost',True);root.update();time.sleep(1.5);root.update()
    window=None
    if not args.verify_ui and sys.platform=='darwin':
        from create_tutorial import application_window_id
        window=application_window_id()
    timeline=[];scene=0;scene_start=0;last_path=None;frame_number=0
    OUT_W,OUT_H=1280,800;CAP_H=100
    def load_font(names,size):
        for n in names:
            try:return ImageFont.truetype(n,size)
            except OSError:pass
        return ImageFont.load_default()
    font=load_font(['/System/Library/Fonts/Supplemental/Arial.ttf','C:/Windows/Fonts/arial.ttf','DejaVuSans.ttf'],24)
    bold=load_font(['/System/Library/Fonts/Supplemental/Arial Bold.ttf','C:/Windows/Fonts/arialbd.ttf','DejaVuSans-Bold.ttf'],26)
    def crop_box_points():
        # Crop away the left controls: keep from the plot's left edge to the window's right edge.
        left=app.scatter_fig.canvas.get_tk_widget().winfo_rootx()-root.winfo_rootx()
        return max(0,left-4)
    def capture(duration=.125,pointer=None,click=False):
        nonlocal frame_number,last_path
        root.update()
        if args.verify_ui:timeline.append((None,duration));return
        if sys.platform=='darwin':
            raw=frames/'window-capture.png'
            for attempt in range(20):
                if subprocess.run(['screencapture','-x','-o','-l',str(window),str(raw)],capture_output=True).returncode==0:break
                root.lift();root.update();time.sleep(.5)
            else:raise RuntimeError('screencapture could not capture the application window')
            base=Image.open(raw).convert('RGB')
            scale=base.width/root.winfo_width();title_bar=base.height-root.winfo_height()*scale
        else:
            from PIL import ImageGrab
            x,y=root.winfo_rootx(),root.winfo_rooty()
            base=ImageGrab.grab(bbox=(x,y,x+root.winfo_width(),y+root.winfo_height()),all_screens=True).convert('RGB')
            scale=base.width/root.winfo_width();title_bar=0  # grab starts at the client area
        left=int(crop_box_points()*scale);top=int(title_bar)
        content=base.crop((left,top,base.width,base.height))
        cw=OUT_W;ch=int(content.height*cw/content.width)
        canvas=Image.new('RGB',(OUT_W,ch+CAP_H),'#102d37');canvas.paste(content.resize((cw,ch)),(0,0))
        draw=ImageDraw.Draw(canvas)
        if pointer is not None:
            px=(pointer[0]*scale-left)*cw/content.width;py=pointer[1]*scale*ch/content.height
            draw.polygon([(px,py),(px+1,py+29),(px+8,py+22),(px+15,py+35),(px+21,py+32),(px+14,py+20),(px+25,py+19)],fill='white',outline='#17212b',width=2)
        draw.text((24,ch+12),entries[scene]['title'],fill='white',font=bold)
        draw.text((24,ch+52),textwrap.fill(entries[scene]['short'],95),fill='#d6e5e8',font=font)
        last_path=frames/f'v2_{frame_number:04d}.png';frame_number+=1
        canvas.resize((OUT_W,(ch+CAP_H)//2*2)).save(last_path);timeline.append((last_path,duration))
    def finish_scene():
        nonlocal scene,scene_start
        remaining=entries[scene]['duration']-sum(d for _,d in timeline[scene_start:])
        if remaining<0:raise RuntimeError(f'Scene {scene+1} animation exceeds its narration chapter by {-remaining:.1f}s')
        if remaining:timeline.append((last_path,remaining))
        scene+=1;scene_start=len(timeline)
    def event(xy):
        x,y=app.ax.transData.transform(xy)
        return SimpleNamespace(inaxes=app.ax,x=float(x),y=float(y),xdata=float(xy[0]),ydata=float(xy[1]),dblclick=False)
    def plot_pointer(xy):
        x,y=app.ax.transData.transform(xy);w=app.scatter_fig.canvas.get_tk_widget()
        return (w.winfo_rootx()-root.winfo_rootx()+x,w.winfo_rooty()-root.winfo_rooty()+w.winfo_height()-y)
    def widget_point(w):
        root.update();return (w.winfo_rootx()-root.winfo_rootx()+w.winfo_width()/2,w.winfo_rooty()-root.winfo_rooty()+w.winfo_height()/2)
    def drag_path(start,target,press,motion,release,steps=10):
        capture(.3,plot_pointer(start));assert press(event(start)) is not False
        capture(.2,plot_pointer(start),True)
        for t in np.linspace(0,1,steps+1)[1:]:
            xy=start*(1-t)+target*t;motion(event(xy));capture(.125,plot_pointer(xy),True)
        release(event(target));capture(.35,plot_pointer(target))
    def drag_handle(kind,c,target):
        h=next(h for h in app.llm_handles if h['kind']==kind and h['class']==c)
        if kind=='arrow':start=h['end'].copy()
        else:
            start=None
            for a in np.linspace(0,2*np.pi,24,endpoint=False):
                pt=h['center']+h['radius']*np.array([np.cos(a),np.sin(a)]);hit=_llm_handle_at(app,event(pt))
                if hit is not None and hit['kind']==kind and hit['class']==c:start=pt;break
            if start is None:raise RuntimeError('Could not grab the intended circle')
        drag_path(start,target,lambda e:llm_overlay_press(app,e),lambda e:llm_overlay_motion(app,e),lambda e:llm_overlay_release(app,e))
    from matplotlib.backend_bases import MouseEvent
    def mouse(name):
        def fire(e):
            MouseEvent(name,app.scatter_fig.canvas,e.x,e.y,button=1)._process();return True
        return fire
    def editor_widget():
        def walk(w):
            if isinstance(w,tk.Text):return w
            for ch in w.winfo_children():
                f=walk(ch)
                if f is not None:return f
        return walk(app.llm_panel.suggestion_frame)

    panel=app.llm_panel
    import ui.ui_llm as panel_module
    panel_module.messagebox.askyesno=lambda *a,**k:True
    def find_widget(root_widget,pred):
        if pred(root_widget):return root_widget
        for ch in root_widget.winfo_children():
            f=find_widget(ch,pred)
            if f is not None:return f
    def button(text):
        def has(w):
            try:return w.cget('text')==text and 'Button' in w.winfo_class()
            except tk.TclError:return False
        w=find_widget(panel,has)
        if w is None:raise RuntimeError(f'Button not found: {text}')
        return w
    def checkbox(text):
        def has(w):
            try:return w.cget('text')==text and w.winfo_class()=='TCheckbutton'
            except tk.TclError:return False
        w=find_widget(panel,has)
        if w is None:raise RuntimeError(f'Checkbox not found: {text}')
        return w
    def toggle_box(text,hold=.5):
        w=checkbox(text);pt=widget_point(w);capture(hold,pt);capture(.3,pt,True);w.invoke();root.update();capture(hold,pt)
    def point_at(widget,hold=1.5):
        capture(hold,widget_point(widget))
    def click(text,hold=.6):
        b=button(text);pt=widget_point(b);capture(hold,pt);capture(.4,pt,True);b.invoke();root.update()
    def wait_interpretation(s,limit=30):
        t0=time.monotonic()
        while panel.text_interpretations.get(s.uid,(None,'pending'))[1]=='pending':
            root.update();time.sleep(.05)
            if time.monotonic()-t0>limit:raise RuntimeError('Text interpretation timed out')
        if panel.text_interpretations[s.uid][1]!='ready':raise RuntimeError('Text interpretation failed')
    def type_text(widget,pt,text):
        capture(.6,pt);widget.focus_set();capture(.25,pt,True);widget.delete('1.0',tk.END)
        for i in range(0,len(text),4):
            widget.insert(tk.END,text[i:i+4]);widget.see(tk.END);capture(.125,pt)
    from llm import openrouter
    replies={}
    def fake_completion(*a,**k):
        blob=json.dumps(a)+json.dumps(k,default=str)
        for key,raw in replies.items():
            if key in blob:return raw,{}
        raise RuntimeError('No cached tutorial interpretation for this text')
    openrouter.chat_completion=fake_completion

    # 0: intro, 1: legend and training log
    capture(2.);finish_scene()
    legend=find_widget(panel,lambda w:w.winfo_class()=='TLabel' and w.cget('text')=='Plot legend')
    point_at(legend,2.5);point_at(panel.training_log,2.5);capture(.5,widget_point(panel.training_log));finish_scene()
    # 2: Get Suggestions, replaying the recorded response through the real button
    focus_entry=find_widget(panel.focus_row,lambda w:w.winfo_class()=='TEntry')
    point_at(focus_entry,1.2)
    orig=panel_module.request_suggestions
    panel_module.request_suggestions=lambda *a,**kw:(summary,suggestions,{},record['raw'])
    b=button('Get Suggestions');capture(.8,widget_point(b));capture(.4,widget_point(b),True);b.invoke()
    t0=time.monotonic()
    while panel.request_running:
        root.update();time.sleep(.05)
        if time.monotonic()-t0>20:raise RuntimeError('Suggestions response timed out')
    panel_module.request_suggestions=orig
    root.update();capture(1.5);finish_scene()
    # 3: the Arrow / Tighten boxes, Previous / Next and Highlight
    toggle_box('Tighten');toggle_box('Tighten')
    toggle_box('Arrow');toggle_box('Arrow')
    point_at(panel.prev_button,.8);point_at(panel.skip_button,.8)
    hl=checkbox('Highlight suggested classes')
    if 0<widget_point(hl)[1]-(panel.canvas.winfo_rooty()-root.winfo_rooty())<panel.canvas.winfo_height():toggle_box('Highlight suggested classes');toggle_box('Highlight suggested classes')
    capture(.5);finish_scene()
    # 4: arrows (direction, distance) and circle (tightness)
    s=app.llm_focus_suggestion
    h=next(h for h in app.llm_handles if h['kind']=='arrow' and h['class']==s.class_i)
    drag_handle('arrow',s.class_i,h['start']+.7*(h['end']-h['start']))
    h=next(h for h in app.llm_handles if h['kind']=='arrow' and h['class']==s.class_j)
    v=h['end']-h['start'];rot=np.array([[.94,-.34],[.34,.94]])
    drag_handle('arrow',s.class_j,h['start']+rot@v*1.15)
    circles=[h for h in app.llm_handles if h['kind']=='circle']
    if circles:
        h=circles[0];drag_handle('circle',h['class'],h['center']+np.array([1.4*h['radius'],0.]))
    capture(.6);finish_scene()
    # 5: edit the text and press "Update plot from instruction"
    instruction='Move cat a little away from dog. Tighten only cat.'
    replies[instruction]=CACHED_TEXT_REPLY
    editor=editor_widget();pt=(editor.winfo_rootx()-root.winfo_rootx()+40,editor.winfo_rooty()-root.winfo_rooty()+22)
    type_text(editor,pt,instruction)
    click('Update plot from instruction');wait_interpretation(s);root.update();capture(1.2);finish_scene()
    # 6: review all five: accept, accept, reject, tweak+accept, reject
    click('Accept');capture(.9)
    click('Accept');capture(.9)
    click('Reject');capture(.9)
    s4=app.llm_focus_suggestion
    h=next((h for h in app.llm_handles if h['kind']=='arrow' and h['class']==s4.class_i),None)
    if h is not None:drag_handle('arrow',s4.class_i,h['start']+.6*(h['end']-h['start']))
    click('Accept');capture(.9)
    click('Reject');capture(1.);finish_scene()
    # 7: add my own suggestion
    click('Add my own',hold=.8)
    def combos():
        out=[]
        def walk(w):
            if w.winfo_class()=='TCombobox':out.append(w)
            for ch in w.winfo_children():walk(ch)
        walk(panel.suggestion_frame);return [c for c in out if c.winfo_ismapped()]
    cn=panel._class_names()
    def pick(slot,name):
        c=combos()[slot];pt=widget_point(c);capture(.4,pt);capture(.25,pt,True)
        c.set(name);c.event_generate('<<ComboboxSelected>>');root.update();capture(.4,pt)
    pick(1,cn[9]);pick(0,cn[1])
    own=app.llm_focus_suggestion
    custom_text=f'Move {cn[1]} away from {cn[9]}. Tighten only {cn[1]}.'
    replies[custom_text]=json.dumps(dict(class_i=1,class_j=9,direction='away_from_j',scale='medium',tighten_i=True,tighten_j=False))
    editor=editor_widget();pt=(editor.winfo_rootx()-root.winfo_rootx()+40,editor.winfo_rooty()-root.winfo_rooty()+22)
    type_text(editor,pt,custom_text)
    click('Update plot from instruction');wait_interpretation(own);root.update();capture(.8)
    h=next(h for h in app.llm_handles if h['kind']=='arrow' and h['class']==own.class_i)
    drag_handle('arrow',own.class_i,h['start']+.8*(h['end']-h['start']))
    click('Accept');capture(1.);finish_scene()
    # 8: Continue Training is shown, not run
    b=button('Continue Training');capture(1.,widget_point(b));capture(2.,widget_point(b));finish_scene()
    if args.verify_ui:
        app.on_close();print('v2 UI walkthrough passed without capture or API calls.');return
    manifest=frames/'v2_frames.txt'
    manifest.write_text(''.join(f"file '{Path(p).as_posix()}'\nduration {d:.6f}\n" for p,d in timeline)+f"file '{Path(timeline[-1][0]).as_posix()}'\n")
    encode(entries,output,frames,args.name);print(f'{frame_number} frames captured',flush=True)
    app.on_close()
if __name__=='__main__':main()
