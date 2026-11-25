'use strict';
const $=id=>document.getElementById(id), KEY='hitl-local-survey-v1';
let startingNewParticipant=false;
let handoffPage=null, handoffSequence=0, handoffTimer;
function handoffScenario(pages,sourcePage){
 const next=pages[sourcePage+1]?.ids;
 if(next?.includes('QID1217253034'))return 'A';
 if(next?.includes('QID185'))return 'B';
}
async function requestTool(){
 const sourcePage=page;
 const scenario=handoffScenario(pages,sourcePage);
 if(!scenario)throw Error('Could not identify the next scenario.');
 const response=await fetch('/api/handoff',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({study_id:state.study_id,session_id:state.session_id,page:sourcePage,scenario})});
 if(!response.ok)throw Error((await response.json()).error||'Could not connect to tool');
 handoffSequence=(await response.json()).sequence;
 handoffPage=sourcePage;
 state.events.push({type:'survey_tool_handoff',page,scenario,sequence:handoffSequence,time:new Date().toISOString()});
 saveNow().catch(()=>{});
}
function toolHandoff(){
 clearInterval(handoffTimer);
 if(!pages[page]?.ids.includes('QID175'))return;
 const scenarioTitle=pages[page+1]?.title ?? 'this scenario';
 const note=element('p',`Preparing the tool for ${scenarioTitle}. Keep this survey tab open; it will continue when the task finishes.`);
 const retry=element('button','Open tool again',{type:'button'});
 retry.onclick=()=>requestTool().catch(e=>note.textContent=e.message);
 $('questions').append(note,retry);
 requestTool().catch(e=>note.textContent=e.message);
 handoffTimer=setInterval(async()=>{
  try{
   const r=await fetch('/api/tool?'+new URLSearchParams({study_id:state.study_id}));
   const status=await r.json();
   if(status.sequence!==handoffSequence)return;
   if(status.state==='active')note.textContent=`${scenarioTitle} is active in the tool. This survey will continue when you finish.`;
   if(status.state==='error')note.textContent=status.reason;
   if(status.state==='complete'&&page===handoffPage){
    clearInterval(handoffTimer);
    state.events.push({type:'tool_survey_handoff',page,sequence:handoffSequence,time:new Date().toISOString()});
    await saveNow();
    page++;render();saveLocal();window.focus();
   }
  }catch{}
 },1000);
}
let schema,surveySha256,state,page=0,pages=[],saveTimer,saveQueue=Promise.resolve(),recorder,recordStreams=[],recordChunks=[],uploadQueue=Promise.resolve(),recordId,part=0,recordingSession;
function element(tag,text,attrs={}){const e=document.createElement(tag);if(text!==null)e.textContent=text;Object.assign(e,attrs);return e;}
function fieldKey(q,row){return row?`${q.id}:${row}`:q.id;}
function makePages(){
 const {beforeScenarios,scenarioOrders,afterScenarios}=schema.flow;
 const scenarioBlocks=scenarioOrders[state.answers.QID176] ?? [];
 return [...beforeScenarios,...scenarioBlocks,...afterScenarios].flatMap(name=>{
  const {title,pages}=schema.blocks[name];
  return pages.map(ids=>({title,ids}));
 });
}
function scenarioAnswers(usabilityId,workloadIds,trustId,toolId){
 const raw=state.answers;
 const usability=schema.questions[usabilityId];
 const agreement=row=>usability.answers.find(option=>option.id===raw[fieldKey(usability,row)])?.text;
 const [mentalDemand,physicalDemand,temporalDemand,performance,effort,frustration]=workloadIds.map(id=>raw[id+':1']);
 const trustScores=schema.questions[trustId].choices.map(row=>{
  const value=raw[fieldKey(schema.questions[trustId],row.id)];
  return value===undefined?undefined:Number(value);
 });
 const [confidence,reliability,trust]=trustScores;
 return {
  usability_agreement:{
   meets_requirements:agreement('1'),
   easy_to_use:agreement('3')
  },
  workload_1_to_20:{
   mental_demand:mentalDemand,
   physical_demand:physicalDemand,
   temporal_demand:temporalDemand,
   performance,
   effort,
   frustration
  },
  s_tias_1_to_7:{
   confidence,
   reliability,
   trust,
   mean:trustScores.every(Number.isFinite)?(confidence+reliability+trust)/3:undefined
  },
  tool_ratings_0_to_100:{
   helped_find_optimal_model:raw[toolId+':1'],
   distracting:raw[toolId+':9'],
   clarified_model_effectiveness:raw[toolId+':10'],
   introduced_bias:raw[toolId+':12'],
   trust_in_model:raw[toolId+':13'],
   understood_visualization:raw[toolId+':14'],
   clarified_model_weaknesses:raw[toolId+':5']
  }
 };
}
function snapshot(){
 const raw=state.answers;
 const selectedLabel=id=>schema.questions[id].choices.find(choice=>choice.id===raw[id])?.text;
 const answers={
  participant:{
   study_id:state.study_id,
   gender:selectedLabel('QID7'),
   gender_other:raw.QID7==='5'?raw['QID7:5:text']:undefined,
   age:raw.QID9,
   models_created_and_trained:raw.QID174,
   ml_experience_0_to_100:raw['QID10:1']
  },
  scenario_order:selectedLabel('QID176'),
  scenario_a:scenarioAnswers('QID1217253034',['QID163','QID164','QID165','QID166','QID167','QID168'],'QID193','QID133'),
  scenario_b:scenarioAnswers('QID185',['QID186','QID187','QID188','QID189','QID190','QID191'],'QID194','QID192'),
  experimenter_notes:raw.QID173
 };
 return {...state,answers,response_format_version:3,page,survey_sha256:surveySha256,updated_at:new Date().toISOString()};
}
function saveLocal(){localStorage.setItem(KEY,JSON.stringify(snapshot()));}
function saveNow(){saveLocal();const data=JSON.stringify(snapshot());saveQueue=saveQueue.catch(()=>{}).then(async()=>{const r=await fetch('/api/responses',{method:'POST',headers:{'Content-Type':'application/json'},body:data});if(!r.ok)throw Error('Local server could not save');$('saveStatus').textContent='Saved on this computer';});return saveQueue.catch(e=>{$('saveStatus').textContent='Server save failed; browser backup retained';throw e;});}
function changed(key,value){state.answers[key]=value;if(key==='QID24')state.study_id=value;state.events.push({type:'answer_changed',question:key,time:new Date().toISOString()});saveLocal();clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveNow().catch(()=>{}),400);}
function inputFor(q,key,container,type='text'){const input=element(type==='textarea'?'textarea':'input',null,{type:type==='textarea'?undefined:type,value:state.answers[key]??''});input.dataset.answer=key;input.required=q.required;input.addEventListener('input',()=>changed(key,input.value));container.append(input);return input;}
function renderQuestion(q){const box=element('fieldset',null),legend=element('legend',q.text+(q.required?' *':''));box.append(legend);if(q.type==='DB'){legend.style.fontWeight='normal';return box;}
 if(q.type==='TE'){inputFor(q,q.id,box,q.selector==='ML'?'textarea':'text');}
 else if(q.type==='MC'){for(const c of q.choices){const row=element('label',null,{className:'choice'}),input=element('input',null,{type:q.selector?.startsWith('MA')?'checkbox':'radio',name:q.id,value:c.id,required:q.required});input.checked=Array.isArray(state.answers[q.id])?state.answers[q.id].includes(c.id):String(state.answers[q.id])===c.id;input.addEventListener('change',()=>{if(input.type==='checkbox'){const vals=[...box.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value);changed(q.id,vals);}else changed(q.id,c.id);});row.append(input,element('span',c.text));box.append(row);if(c.textEntry){const extra=inputFor({...q,required:false},`${q.id}:${c.id}:text`,box);extra.placeholder='Please specify';}}}
 else if(q.type==='Matrix'){for(const c of q.choices){const row=element('div',null,{className:'matrix-row'});row.append(element('div',c.text));const scale=element('div',null,{className:'scale'});for(const a of q.answers){const label=element('label',null),input=element('input',null,{type:'radio',name:fieldKey(q,c.id),value:a.id,required:q.required});input.checked=String(state.answers[fieldKey(q,c.id)])===a.id;input.addEventListener('change',()=>changed(fieldKey(q,c.id),a.id));label.append(input,element('span',a.text));scale.append(label);}row.append(scale);box.append(row);}}
 else if(q.type==='Slider'){for(const c of q.choices){const key=fieldKey(q,c.id),row=element('div',null),label=element('label',c.text,{className:'slider-label'});const input=element('input',null,{type:'range',min:q.min,max:q.max,step:'1',value:state.answers[key]??q.min});input.setAttribute('aria-label',c.text||q.text);input.dataset.answer=key;const output=element('output',state.answers[key]??'Choose a value');input.addEventListener('input',()=>{changed(key,Number(input.value));output.textContent=input.value;});label.append(input,output);const ends=element('div',null,{className:'slider-endpoints'});ends.append(element('span',q.labels[0]||String(q.min)),element('span',q.labels.at(-1)||String(q.max)));row.append(label,ends);box.append(row);}}
 return box;}
function render(){clearInterval(handoffTimer);pages=makePages();page=Math.max(0,Math.min(page,pages.length-1));const current=pages[page];$('questions').replaceChildren(...current.ids.map(id=>renderQuestion(schema.questions[id])));$('pageTitle').textContent=current.title;$('pagePosition').textContent=`${page+1} / ${pages.length}`;$('previous').disabled=page===0;$('next').textContent=page===pages.length-1?'Save and finish':'Save and continue';$('validation').textContent='';state.events.push({type:'page_view',page,block:current.title,time:new Date().toISOString()});toolHandoff();}
function validate(){if(!$('questions').reportValidity())return false;for(const id of pages[page].ids){const q=schema.questions[id];if(q.required&&q.type==='Slider'){for(const c of q.choices)if(state.answers[fieldKey(q,c.id)]===undefined){$('validation').textContent='Please choose a value for every slider, including values at the starting position.';return false;}}}return true;}
function showTutorialError(){
 $('videoNote').textContent='The tutorial video could not load.';
}
$('tutorialVideo').addEventListener('error',showTutorialError);
function updateBeginAvailability(){
 const recordingActive=recorder?.state==='recording';
 $('begin').disabled=!recordingActive;
 $('beginHint').hidden=recordingActive;
}
$('begin').onclick=()=>{if(!state||recorder?.state!=='recording')return;if(!$('studyId').value.trim()){$('studyId').reportValidity();return;}state.study_id=$('studyId').value.trim();state.answers.QID24=state.study_id;$('setup').hidden=true;$('beginControls').hidden=true;$('survey').hidden=false;render();saveNow().catch(()=>{});};
$('next').onclick=async()=>{if(!validate())return;try{await saveNow();}catch{return;}pages=makePages();if(page===pages.length-1){state.completed_at=new Date().toISOString();await saveNow();$('survey').hidden=true;$('finished').hidden=false;setRecordingControls(false);}else{page++;render();saveLocal();}};
$('previous').onclick=()=>{page--;render();saveLocal();};
$('downloadBackup').onclick=()=>{
 const recordingBackup=$('downloadRecording');
 if(!recordingBackup.hasAttribute('href'))return;
 const answersUrl=URL.createObjectURL(new Blob([JSON.stringify(snapshot(),null,2)],{type:'application/json'}));
 const answersBackup=element('a',null,{href:answersUrl,download:`${state.study_id}-answers.json`});
 recordingBackup.click();
 answersBackup.click();
 setTimeout(()=>URL.revokeObjectURL(answersUrl),1000);
};
$('newSession').onclick=()=>{if($('recordStart').disabled){alert('Stop and save the recording first.');return;}startingNewParticipant=true;location.reload();};
function setRecordingControls(compact){
 const panel=$('recording'),stop=$('recordStop');
 const finished=!$('finished').hidden;
 const destination=$(finished?'finishRecording':compact?'headerControls':'recordActions');
 if(stop.parentElement!==destination){
  destination.append(stop);
  if(finished)destination.append($('recordStatus'));
 }
 stop.textContent=compact&&!finished?'Stop recording':'Stop and save recording';
 stop.hidden=finished&&stop.disabled;
 panel.hidden=compact||finished;
}

$('recordStart').onclick=async()=>{if(!$('consent').checked){$('recordStatus').textContent='Participant agreement is required before recording.';return;}
 const start=$('recordStart'),status=$('recordStatus'),microphone=$('recordMicrophone');
 if(!state){status.textContent='Wait for the survey to load before recording.';return;}
 const studyId=state.study_id||$('studyId').value.trim();
 if(!studyId){status.textContent='Enter the Study ID before recording.';return;}
 const includeMicrophone=microphone.checked;
 start.disabled=true;
 microphone.disabled=true;
 $('studyId').disabled=true;
 let stage='screen';
 try{state.events.push({type:'recording_consent_confirmed',microphone:includeMicrophone,time:new Date().toISOString()});const screen=await navigator.mediaDevices.getDisplayMedia({video:true,audio:false});recordStreams=[screen];
 if(includeMicrophone){
  stage='microphone';
  status.textContent='Waiting for microphone access...';
  recordStreams.push(await navigator.mediaDevices.getUserMedia({audio:true}));
 }
 stage='recorder';
 const combined=new MediaStream(recordStreams.flatMap(stream=>stream.getTracks()));const mime=['video/mp4','video/webm;codecs=vp8,opus','video/webm'].find(m=>MediaRecorder.isTypeSupported(m));recorder=new MediaRecorder(combined,mime?{mimeType:mime}:{});recordId=crypto.randomUUID();recordingSession=state.session_id;part=0;recordChunks=[];uploadQueue=Promise.resolve();let uploadFailed=false,savedFilename;
 recorder.ondataavailable=e=>{if(!e.data.size)return;recordChunks.push(e.data);const currentPart=part++;uploadQueue=uploadQueue.then(async()=>{const query=new URLSearchParams({session:recordingSession,recording:recordId,study_id:studyId,part:currentPart,mime:recorder.mimeType});const r=await fetch('/api/recordings?'+query,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:e.data});if(!r.ok)throw Error('Recording upload failed');savedFilename=(await r.json()).filename;}).catch(()=>{uploadFailed=true;$('recordStatus').textContent='Local recording save failed. Stop and download the backup before closing.';});};
 recorder.onstop=async()=>{updateBeginAvailability();for(const stream of recordStreams)stream.getTracks().forEach(t=>t.stop());await uploadQueue;const blob=new Blob(recordChunks,{type:recorder.mimeType});const backup=$('downloadRecording');backup.href=URL.createObjectURL(blob);backup.download=savedFilename||`${studyId}-${recordId.slice(0,8)}.${recorder.mimeType.includes('mp4')?'mp4':'webm'}`;backup.hidden=false;$('downloadBackup').hidden=false;$('recordStatus').textContent=uploadFailed?'Save failed: download the recording backup.':'Recording saved locally. A backup download is also available.';start.disabled=false;microphone.disabled=false;$('studyId').disabled=false;$('recordStop').disabled=true;$('recording').classList.remove('recording');setRecordingControls(false);state.recordings.push({id:recordId,mime:recorder.mimeType,microphone:includeMicrophone,parts:part,save_failed:uploadFailed,stopped_at:new Date().toISOString()});saveNow().catch(()=>{});};
 screen.getVideoTracks()[0].onended=()=>{if(recorder.state==='recording'){recorder.stop();updateBeginAvailability();}};recorder.start(2000);updateBeginAvailability();$('recordStart').disabled=true;$('recordStop').disabled=false;status.textContent=includeMicrophone?'Recording screen and microphone.':'Recording screen only.';$('recording').classList.add('recording');setRecordingControls($('finished').hidden);
 }catch(e){
  recordStreams.forEach(stream=>stream.getTracks().forEach(track=>track.stop()));
  recordStreams=[];
  start.disabled=false;
  microphone.disabled=false;
  $('studyId').disabled=false;
  updateBeginAvailability();
  if(stage==='screen'&&e.name==='NotFoundError'){
   status.textContent='Chrome could not find a screen or window to share. If you are connected remotely, try recording from Chrome on the PC itself.';
  }else if(stage==='microphone'&&e.name==='NotFoundError'){
   status.textContent='No microphone was found. Turn off Include microphone audio for a screen-only recording, or connect a microphone and try again.';
  }else if(stage==='microphone'&&e.name==='NotAllowedError'){
   status.textContent='Microphone access was blocked. Allow it in your browser and system privacy settings, then start recording again.';
  }else{
   const action={screen:'share the screen',microphone:'open the microphone',recorder:'start the recorder'}[stage];
   status.textContent='Could not '+action+': '+e.message;
  }
 }
};
$('recordStop').onclick=()=>{if(recorder?.state==='recording'){if($('finished').hidden&&$('recording').hidden&&!window.confirm('Stop recording? Your questionnaire will stay open.'))return;recorder.stop();updateBeginAvailability();$('recordStop').disabled=true;setRecordingControls(false);$('recordStatus').textContent='Saving recording…';}};
window.addEventListener('beforeunload',e=>{
 if(startingNewParticipant)return;
 e.preventDefault();
 e.returnValue='';
});
(async()=>{
 const response=await fetch('survey.json', {cache:'no-cache'});
 if(!response.ok)throw Error(`Survey definition returned HTTP ${response.status}`);
 const bytes=await response.arrayBuffer();
 schema=JSON.parse(new TextDecoder().decode(bytes));
 const hash=await crypto.subtle.digest('SHA-256',bytes);
 surveySha256=Array.from(new Uint8Array(hash),byte=>byte.toString(16).padStart(2,'0')).join('');
 state={session_id:crypto.randomUUID(),study_id:'',answers:{},events:[],recordings:[],started_at:new Date().toISOString()};
 updateBeginAvailability();
 if($('tutorialVideo').error)showTutorialError();
 $('studyId').value=new URLSearchParams(location.search).get('studyId')||'';
 $('consent').checked=false;
 $('recordMicrophone').checked=$('recordMicrophone').defaultChecked;
 $('saveStatus').textContent='Ready — saves locally';
})().catch(e=>{$('saveStatus').textContent='Could not load survey: '+e.message;});

for(const name of ['focus','blur'])window.addEventListener(name,()=>{
 if(!state)return;
 state.events.push({type:'survey_focus_changed',focused:name==='focus',page,time:new Date().toISOString()});
 saveNow().catch(()=>{});
});
