const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/web/app.js','utf8').split("$('begin').onclick")[0];
let poll,posted=[],status={sequence:1,state:'active'},focused=0;
const node=()=>({append(){},addEventListener(){},textContent:''});
const context=vm.createContext({
 document:{getElementById:node,createElement:node},console,setTimeout,clearTimeout,
 setInterval:fn=>(poll=fn,1),clearInterval(){},URLSearchParams,
 window:{focus(){focused++;}},
 fetch:async(url,options)=>{
  if(url==='/api/handoff'){posted.push(JSON.parse(options.body));return {ok:true,json:async()=>({sequence:1})};}
  return {ok:true,json:async()=>status};
 }
});
vm.runInContext(source,context);
vm.runInContext(`state={study_id:'P01',session_id:'survey-1',events:[]};page=2;pages=[{},{},{ids:['QID175']}];
 saveNow=async()=>{};saveLocal=()=>{};render=()=>{};toolHandoff();`,context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(posted.length,1);assert.equal(posted[0].study_id,'P01');assert.equal(posted[0].page,2);
 await poll();assert.equal(vm.runInContext('page',context),2);
 status={sequence:99,state:'complete'};await poll();assert.equal(vm.runInContext('page',context),2);
 status={sequence:1,state:'complete'};await poll();assert.equal(vm.runInContext('page',context),3);
 assert.equal(focused,1);
 assert.deepEqual(Array.from(vm.runInContext('state.events.map(e=>e.type)',context)),['survey_tool_handoff','tool_survey_handoff']);
 console.log('Handoff checks passed: matching Study ID, stale completion ignored, completion advances once and is logged.');
})().catch(error=>{console.error(error);process.exitCode=1;});
