"""Capture a captioned tutorial of the actual desktop UI using a saved CIFAR10 model.
Uses public CIFAR10 demo data and one backend LLM interpretation, or a cached reply.
Run from repository root on macOS.
"""
import os,sys,time,json,subprocess,argparse,math,wave,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));os.chdir(ROOT)
import tkinter as tk
import numpy as np
from types import SimpleNamespace
import torch
from PIL import Image,ImageDraw,ImageFont
from data.data_loader import load_dataset
from model import get_model
from ui.ui import UI
from llm.suggestions import parse_suggestions
from llm.strategies import STRATEGIES
from ui.ui_display import (refresh_llm_overlay, llm_overlay_press, llm_overlay_motion,
                           llm_overlay_release, _llm_handle_at)
from llm.movement import fit_independent_class_vector


def application_window_id():
    cg=ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
    cf=ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    cg.CGWindowListCopyWindowInfo.argtypes=[ctypes.c_uint32,ctypes.c_uint32]
    cg.CGWindowListCopyWindowInfo.restype=ctypes.c_void_p
    cf.CFArrayGetCount.argtypes=[ctypes.c_void_p];cf.CFArrayGetCount.restype=ctypes.c_long
    cf.CFArrayGetValueAtIndex.argtypes=[ctypes.c_void_p,ctypes.c_long];cf.CFArrayGetValueAtIndex.restype=ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes=[ctypes.c_void_p,ctypes.c_void_p];cf.CFDictionaryGetValue.restype=ctypes.c_void_p
    cf.CFNumberGetValue.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    cf.CFStringGetCString.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_long,ctypes.c_uint32]
    cf.CFRelease.argtypes=[ctypes.c_void_p]
    keys={k:ctypes.c_void_p.in_dll(cg,k).value for k in ['kCGWindowOwnerPID','kCGWindowNumber','kCGWindowName']}
    windows=cg.CGWindowListCopyWindowInfo(0,0)
    try:
        for i in range(cf.CFArrayGetCount(windows)):
            item=cf.CFArrayGetValueAtIndex(windows,i)
            value=cf.CFDictionaryGetValue(item,keys['kCGWindowOwnerPID'])
            pid=ctypes.c_int32()
            if not value:continue
            cf.CFNumberGetValue(value,3,ctypes.byref(pid))
            if pid.value!=os.getpid():continue
            name=cf.CFDictionaryGetValue(item,keys['kCGWindowName'])
            buffer=ctypes.create_string_buffer(1024)
            if name:cf.CFStringGetCString(name,buffer,1024,0x08000100)
            if b'HITL-ML' not in buffer.value:continue
            number=ctypes.c_int32();cf.CFNumberGetValue(cf.CFDictionaryGetValue(item,keys['kCGWindowNumber']),3,ctypes.byref(number))
            return number.value
    finally:
        cf.CFRelease(windows)
    raise RuntimeError('Could not identify this application window; refusing a desktop capture.')


NARRATION = [
('1. Explore the representation', 'This is the latent space view. Each dot is an example, and colours show its class. The suggestions sidebar is already open on the right.'),
('2. Get suggestions', 'Click Get Suggestions and wait for the response. Read the proposed changes on the right. Dashed arrows preview the movements. Nothing is applied yet.'),
('3. Focus on the current classes', 'Turn on this option to show only the classes in the current suggestion. Turn it off whenever you need the full picture.'),
('4. Adjust the proposal', 'Watch the cursor drag the first arrow, then the second. Each arrow can be edited independently. Drag the dashed circle to make a group tighter.'),
('5. Edit an instruction', 'Now edit the instruction in the text box. The backend asks the language model to interpret your words. Wait for the updated arrows, then review them.'),
('6. Apply the reviewed change', 'Click Apply when the proposal looks right. The points move, and the log saves your words and the final movements together.'),
('7. Inspect the measured effect', 'These measures describe the proposed change. They show movement size and class separation. A better layout does not always mean better accuracy.'),
('8. Complete a study task', 'Enable Study mode, then click Begin or reset study task. Training stays locked. You have three interactions and five minutes. Watch the count and timer.'),
('9. Finish and return to the survey', 'Collapse either sidebar when you want more space. After the task, return to the survey. Answer the questions, then stop and save the recording.')
]
CAPTIONS = [
'Each dot is an example. Colours identify classes. The suggestions sidebar is ready on the right.',
'Click Get Suggestions. Wait for the response, then review the proposed changes on the right.',
'Use “Only this suggestion’s clusters” to hide unrelated groups while editing.',
'Drag each arrow independently. Drag a circle edge to change how tightly a class is grouped.',
'Edit the text. The backend LLM interprets the instruction and updates the preview automatically.',
'Apply commits the reviewed proposal. The log stores the text and actual movements together.',
'Impact measures describe the proposed layout. They do not guarantee better accuracy.',
'Three interactions, five minutes. Training is disabled during the study task.',
'Return to the survey, answer the questions, then stop and save the screen and audio recording.'
]

def build_narration(frames, output, voice='Daniel', rate=145):
    entries=[]
    for i,(title,text) in enumerate(NARRATION):
        # Short sentences and a small pause between them make the instructions easier to follow.
        spoken=text.replace('. ', '. [[slnc 250]] ')
        raw=frames/f'narration_{i:02d}.aiff';wav=frames/f'narration_{i:02d}.wav'
        subprocess.run(['say','-v',voice,'-r',str(rate),'-o',str(raw),spoken],check=True)
        subprocess.run(['afconvert','-f','WAVE','-d','LEI16@24000',str(raw),str(wav)],check=True)
        with wave.open(str(wav)) as f:seconds=f.getnframes()/f.getframerate()
        if seconds<1:raise RuntimeError('Speech synthesis returned no usable audio.')
        duration=max(12,math.ceil(seconds+1))
        padded=frames/f'audio_{i:02d}.wav'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af','apad','-t',str(duration),str(padded)],check=True)
        entries.append(dict(title=title,caption=text,duration=duration,audio=str(padded)))
    manifest=frames/'audio.txt';manifest.write_text(''.join(f"file '{e['audio']}'\n" for e in entries))
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c:a','aac','-b:a','128k',str(output/'tutorial_narration.m4a')],check=True)
    (output/'tutorial_script.json').write_text(json.dumps([{k:v for k,v in e.items() if k!='audio'} for e in entries],indent=2))
    print('Narration ready:',voice,rate,'words/minute;',sum(e['duration'] for e in entries),'seconds',flush=True)
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--narration-only',action='store_true')
    parser.add_argument('--verify-ui',action='store_true')
    parser.add_argument('--reuse-narration',action='store_true')
    parser.add_argument('--reuse-interpretation',action='store_true',help='replay the saved tutorial LLM reply without another API call')
    parser.add_argument('--voice',default='Daniel')
    parser.add_argument('--rate',type=int,default=145)
    args=parser.parse_args()
    output=ROOT/'study/web';frames=ROOT/'reports/tutorial_frames';frames.mkdir(parents=True,exist_ok=True)
    entries=([dict(title=t,caption=c,duration=12) for t,c in NARRATION] if args.verify_ui else
             json.loads((output/'tutorial_script.json').read_text()) if args.reuse_narration else
             build_narration(frames,output,args.voice,args.rate))
    if args.narration_only:return
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train,val,test,classes,shape=load_dataset('CIFAR10',512)
    model=get_model('CNN_CIFAR10',shape,classes).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    root=tk.Tk();app=UI(root,model,opt,train,val,test,device,'CIFAR10','CNN_CIFAR10','cross_entropy','validation','TUTORIAL','cursor_demo')
    root.geometry('1440x900+10+30');root.update()
    app.load_pretrained_model(str(ROOT/'saved_checkpoints/CIFAR10/checkpoint_epoch_20.pt'))
    # The tutorial begins with the editable suggestions sidebar already enabled.
    app.strategy_display_var.set(STRATEGIES[4].label);app.on_strategy_change();root.update()
    names=dict(enumerate(val.dataset.dataset.classes))
    record=next(json.loads(l) for l in (ROOT/'code/experiments/results/llm_comparison.jsonl').read_text().splitlines()
                if json.loads(l)['model']=='openai/gpt-5-mini' and json.loads(l)['dataset']=='CIFAR10')
    summary,suggestions=parse_suggestions(record['raw'],names,set(names))
    # The tutorial replays a recorded successful suggestions response.
    app.llm_panel.model_var.set('openai/gpt-5-mini')
    for s in suggestions:s.original_text=s.suggestion
    root.update()
    window=None if args.verify_ui else application_window_id()
    timeline=[];scene=0;scene_start=0;last_path=None;last_base=None;frame_number=0
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25)
    except OSError:font=ImageFont.load_default()
    def capture(duration=.125,pointer=None,click=False):
        nonlocal frame_number,last_path,last_base
        root.update()
        if args.verify_ui:
            timeline.append((None,duration));return
        raw=frames/'window-capture.png'
        subprocess.run(['screencapture','-x','-o','-l',str(window),str(raw)],check=True)
        base=Image.open(raw).convert('RGB');last_base=base.size
        canvas=Image.new('RGB',(1440,1040),'#102d37');canvas.paste(base.resize((1440,900)),(0,0))
        draw=ImageDraw.Draw(canvas)
        if pointer is not None:
            # Cursor coordinates are relative to the Tk content; account for the window title bar.
            backing_scale=base.width/root.winfo_width()
            title_bar=base.height-root.winfo_height()*backing_scale
            px=pointer[0]*backing_scale*1440/base.width
            py=(pointer[1]*backing_scale+title_bar)*900/base.height
            if click:draw.ellipse((px-17,py-17,px+17,py+17),fill='#f9cd58',outline='#6b4c00',width=2)
            draw.polygon([(px,py),(px+1,py+29),(px+8,py+22),(px+15,py+35),(px+21,py+32),(px+14,py+20),(px+25,py+19)],fill='white',outline='#17212b',width=2)
        import textwrap
        draw.text((35,922),entries[scene]['title'],fill='white',font=font)
        draw.text((35,961),textwrap.fill(CAPTIONS[scene],105),fill='#d6e5e8',font=font,spacing=8)
        last_path=frames/f'cursor_{frame_number:04d}.png';frame_number+=1;canvas.save(last_path)
        timeline.append((last_path,duration))
    def finish_scene():
        nonlocal scene,scene_start
        elapsed=sum(d for _,d in timeline[scene_start:]);remaining=entries[scene]['duration']-elapsed
        if remaining<0:raise RuntimeError('Demo animation exceeds its narration chapter')
        if remaining:timeline.append((last_path,remaining))
        scene+=1;scene_start=len(timeline)
    def event(xy):
        x,y=app.ax.transData.transform(xy)
        return SimpleNamespace(inaxes=app.ax,x=float(x),y=float(y),xdata=float(xy[0]),ydata=float(xy[1]))
    def plot_pointer(xy):
        x,y=app.ax.transData.transform(xy);widget=app.scatter_fig.canvas.get_tk_widget()
        return (widget.winfo_rootx()-root.winfo_rootx()+x,
                widget.winfo_rooty()-root.winfo_rooty()+widget.winfo_height()-y)
    def drag_handle(kind,c,target):
        h=next(h for h in app.llm_handles if h['kind']==kind and h['class']==c)
        if kind=='arrow':start=h['end'].copy()
        else:
            start=None
            for angle in np.linspace(0,2*np.pi,24,endpoint=False):
                point=h['center']+h['radius']*np.array([np.cos(angle),np.sin(angle)])
                hit=_llm_handle_at(app,event(point))
                if hit is not None and hit['kind']==kind and hit['class']==c:start=point;break
            if start is None:raise RuntimeError('Could not grab the intended circle')
        capture(.25,plot_pointer(start))
        assert llm_overlay_press(app,event(start))
        capture(.25,plot_pointer(start),True)
        for t in np.linspace(0,1,13)[1:]:
            xy=start*(1-t)+target*t
            llm_overlay_motion(app,event(xy))
            capture(.125,plot_pointer(xy),True)
        assert llm_overlay_release(app,event(target))
        capture(.25,plot_pointer(target))
    def editor_widget():
        def walk(widget):
            if isinstance(widget,tk.Text):return widget
            for child in widget.winfo_children():
                found=walk(child)
                if found is not None:return found
        return walk(app.llm_panel.suggestion_frame)
    def widget_point(widget):
        root.update()
        return (widget.winfo_rootx()-root.winfo_rootx()+widget.winfo_width()/2,
                widget.winfo_rooty()-root.winfo_rooty()+widget.winfo_height()/2)
    def find_text(widget,text):
        try:
            if widget.cget('text')==text:return widget
        except tk.TclError:pass
        for child in widget.winfo_children():
            found=find_text(child,text)
            if found is not None:return found
    capture();finish_scene()
    # Exercise the actual Get Suggestions button and asynchronous result path,
    # replaying public demo output instead of paying for another identical request.
    import ui.ui_llm as panel_module
    original_request=panel_module.request_suggestions
    panel_module.request_suggestions=lambda *a,**kw:(summary,suggestions,{},record['raw'])
    button=app.llm_panel.request_button
    capture(1.,widget_point(button));capture(.5,widget_point(button),True)
    button.invoke()
    deadline=time.monotonic()+15
    while app.llm_panel.request_running:
        root.update();time.sleep(.05)
        if time.monotonic()>deadline:raise RuntimeError('Tutorial suggestions response timed out')
    panel_module.request_suggestions=original_request
    capture(1.);finish_scene()
    app.llm_panel.suggestion_view_var.set(True);app.llm_panel._toggle_suggestion_view();capture();finish_scene()
    s=suggestions[0]
    h=next(h for h in app.llm_handles if h['kind']=='arrow' and h['class']==s.class_i)
    drag_handle('arrow',s.class_i,h['start']+.65*(h['end']-h['start']))
    h=next(h for h in app.llm_handles if h['kind']=='arrow' and h['class']==s.class_j)
    vector=h['end']-h['start'];rot=np.array([[.94,-.34],[.34,.94]])
    drag_handle('arrow',s.class_j,h['start']+rot@vector)
    h=next(h for h in app.llm_handles if h['kind']=='circle' and h['class']==s.class_i)
    drag_handle('circle',s.class_i,h['center']+np.array([.35*h['mean_radius'],0.]))
    finish_scene()
    editor=editor_widget()
    if editor is None:raise RuntimeError('Suggestion text editor not found')
    point=(editor.winfo_rootx()-root.winfo_rootx()+40,editor.winfo_rooty()-root.winfo_rooty()+22)
    capture(.75,point);editor.focus_set();capture(.25,point,True)
    editor.delete('1.0',tk.END)
    instruction='Move cat a little away from dog. Tighten only cat.'
    for i in range(0,len(instruction),3):
        editor.insert(tk.END,instruction[i:i+3]);editor.see(tk.END)
        capture(.125,point)
    if args.verify_ui:
        # Verification mode exercises widgets without making an API call.
        fields=dict(class_i=s.class_i,class_j=s.class_j,direction='away_from_j',scale='small',tighten_i=True,tighten_j=False)
        s.suggestion=instruction;s.move_overrides.clear()
        for k,v in fields.items():setattr(s,k,v)
        app.llm_panel.refresh_cards()
    else:
        # Use the same backend validation path when replaying a successful tutorial reply.
        from llm import openrouter
        original_completion=openrouter.chat_completion
        if args.reuse_interpretation:
            cached=json.loads((frames/'cached_interpretation.json').read_text())
            if cached['text']!=instruction:raise RuntimeError('Cached interpretation does not match this tutorial text')
            openrouter.chat_completion=lambda *args,**kwargs:(cached['raw'],{})
        app.llm_panel._set_text(s,instruction)
        capture(1.,point)
        deadline=time.monotonic()+75
        while app.llm_panel.text_interpretations.get(s.uid,(None,'pending'))[1]=='pending':
            root.update()
            if time.monotonic()>deadline:raise RuntimeError('Tutorial text interpretation timed out')
            time.sleep(.05)
        openrouter.chat_completion=original_completion
        if app.llm_panel.text_interpretations[s.uid][1]!='ready':raise RuntimeError('Tutorial text interpretation failed; see application status/log')
        if not args.reuse_interpretation:
            (frames/'cached_interpretation.json').write_text(json.dumps({'text':instruction,'model':'openai/gpt-5-mini',
                'raw':json.dumps(s.interpretation['fields']),'fields':s.interpretation['fields']},indent=2))
        capture(1.)
    finish_scene()
    assert app.llm_panel._apply(s);capture();finish_scene()
    app.llm_panel.current_index=1;app.llm_panel.refresh_cards();capture();finish_scene()
    app.control_panel.master.yview_moveto(.3);root.update()
    checkbox=find_text(app.control_panel,'Study mode (training locked)')
    capture(1.,widget_point(checkbox));capture(.5,widget_point(checkbox),True);checkbox.invoke();root.update()
    begin=find_text(app.control_panel,'Begin / reset study task')
    capture(1.,widget_point(begin));capture(.5,widget_point(begin),True);begin.invoke();root.update()
    capture(1.);finish_scene()
    app.toggle_sidebar('controls');capture();finish_scene()
    if args.verify_ui:
        assert str(app.training_button.cget('state'))=='disabled'
        app.on_close();print('Cursor/text UI walkthrough passed without capture or API calls.');return
    manifest=frames/'frames.txt'
    manifest.write_text(''.join(f"file '{p}'\nduration {d:.6f}\n" for p,d in timeline)+f"file '{timeline[-1][0]}'\n")
    total=sum(e['duration'] for e in entries)
    destination=output/'tutorial_cursor_voice.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-i',str(output/'tutorial_narration.m4a'),
                    '-t',str(total),'-vf','fps=24,format=yuv420p','-c:v','libx264','-preset','fast','-crf','22',
                    '-c:a','aac','-metadata:s:a:0','language=eng','-disposition:a:0','default','-movflags','+faststart',str(destination)],check=True)
    import shutil
    for name in ['tutorial.mp4','tutorial_with_voice.mp4']:shutil.copyfile(destination,output/name)
    def stamp(sec):return f'{sec//3600:02d}:{sec//60%60:02d}:{sec%60:02d}.000'
    start=0;vtt=['WEBVTT\n']
    for i,e in enumerate(entries):vtt.append(f"{i+1}\n{stamp(start)} --> {stamp(start+e['duration'])}\n{e['title']}: {e['caption']}\n");start+=e['duration']
    (output/'tutorial.vtt').write_text('\n'.join(vtt))
    print(f'Tutorial generated: {destination}; {total} seconds; {frame_number} application frames',flush=True)
    app.on_close()
if __name__=='__main__':main()
