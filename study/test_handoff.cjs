const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/web/app.js','utf8').split("$('begin').onclick")[0];
let poll,posted=[],polled=[],status={sequence:1,state:'preparing'},focused=0;
const timers=[];
const nodes=[];
const node=()=>{const element={append(){},addEventListener(){},textContent:''};nodes.push(element);return element;};
const context=vm.createContext({
 document:{getElementById:node,createElement:node},console,
 setTimeout:(fn,ms)=>timers.push({fn,ms}),clearTimeout(){},
 setInterval:fn=>(poll=fn,1),clearInterval(){},URLSearchParams,
 window:{focus(){focused++;}},
 fetch:async(url,options)=>{
  if(url==='/api/handoff'){
   const request=JSON.parse(options.body);posted.push(request);
   return {ok:true,json:async()=>({sequence:request.scenario==='A'?1:2})};
  }
  polled.push(new URLSearchParams(url.split('?')[1]));
  return {ok:true,json:async()=>status};
 }
});
vm.runInContext(source,context);
vm.runInContext(`state={study_id:'P01',session_id:'survey-1',events:[],answers:{}};page=1;pages=[{},{},{ids:['QID175']},{ids:['QID1217253034'],title:'A_Pamap'},{ids:['QID175']},{ids:['QID185'],title:'B_Cifar'}];
 saveNow=async()=>{};saveLocal=()=>{};render=()=>{};makePages=()=>pages;changed('QID176','1');`,context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(posted[0],{study_id:'P01',session_id:'survey-1',page:2,scenario:'A',prepare:true});
 assert.equal(timers.shift().ms,400); // Debounced answer save
 vm.runInContext('page=2;toolHandoff()',context);
 await new Promise(resolve=>setImmediate(resolve));
 const note=nodes.find(element=>element.role==='status');
 await poll();
 assert.equal(note.textContent,'Preparing A_Pamap in the background…');
 assert.equal(timers.length,0);
 assert.equal(posted.length,1);
 status={sequence:1,state:'ready'};
 await poll();
 await poll();
 assert.equal(timers.length,1);
 for(const seconds of [3,2,1]){
  assert.equal(note.textContent,`Switching to A_Pamap in ${seconds}…`);
  assert.equal(posted.length,1);
  const timer=timers.shift();
  assert.equal(timer.ms,1000);
  timer.fn();
 }
 await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(posted[1],{...posted[0],prepare:false});
 status={sequence:1,state:'active'};
 await poll();assert.equal(vm.runInContext('page',context),2);
 assert.equal(polled[0].get('session_id'),'survey-1');
 assert.equal(polled[0].get('scenario'),'A');
 status={sequence:99,state:'complete'};await poll();assert.equal(vm.runInContext('page',context),2);
 status={sequence:1,state:'complete'};await poll();assert.equal(vm.runInContext('page',context),3);
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(focused,1);
 assert.deepEqual(posted[2],{study_id:'P01',session_id:'survey-1',page:4,scenario:'B',prepare:true});
 assert.deepEqual(Array.from(vm.runInContext('state.events.map(e=>e.type)',context)),
  ['answer_changed','survey_tool_prepare','survey_tool_handoff','tool_survey_handoff','survey_tool_prepare']);
 console.log('Handoff checks passed: first and second preload, shared identity, stale completion ignored.');
})().catch(error=>{console.error(error);process.exitCode=1;});
