const fs=require('node:fs');const vm=require('node:vm');const assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/web/app.js','utf8').split("$('begin').onclick")[0];
const context=vm.createContext({document:{getElementById:()=>({addEventListener(){}})},console,setTimeout,clearTimeout});vm.runInContext(source,context);
const schema=JSON.parse(fs.readFileSync(__dirname+'/web/survey.json','utf8'));
function route(order){context.inputSchema=schema;context.order=order;return JSON.parse(vm.runInContext('schema=inputSchema; state={answers:{QID176:order}}; JSON.stringify(makePages())',context));}
for(const [order,expected] of [['1',['A','B']],['2',['B','A']]]){
 context.routePages=route(order);
 const scenarios=JSON.parse(vm.runInContext(
  "JSON.stringify(routePages.flatMap((page,index)=>page.ids.includes('QID175')?[handoffScenario(routePages,index)]:[]))",
  context
 ));
 assert.deepEqual(scenarios,expected);
}
const a=route('1').flatMap(p=>p.ids),b=route('2').flatMap(p=>p.ids);
for(const order of ['1','2']){
 const pages=route(order);
 for(const [workload,trust,tool] of [['QID168','QID193','QID133'],['QID191','QID194','QID192']]){
  const index=pages.findIndex(page=>page.ids.includes(trust));
  assert.deepEqual(pages[index].ids,[trust]);
  assert(pages[index-1].ids.includes(workload));
  assert.deepEqual(pages[index+1].ids,[tool]);
  assert.equal(schema.questions[trust].choices.length,3);
  assert.deepEqual(schema.questions[trust].answers.map(option=>option.id),['1','2','3','4','5','6','7']);
  assert.equal(schema.questions[trust].required,true);
 }
}
assert(a.indexOf('QID1217253034')<a.indexOf('QID185'));
assert(b.indexOf('QID185')<b.indexOf('QID1217253034'));
assert(a.includes('QID173')&&b.includes('QID173'));
assert(!a.includes('QID1')&&!b.includes('QID1'));
for(const [ids,first,second] of [[a,'QID1217253034','QID185'],[b,'QID185','QID1217253034']]){
 const handoffs=ids.flatMap((id,index)=>id==='QID175'?[index]:[]);
 assert.equal(handoffs.length,2);
 assert.equal(ids[handoffs[0]+1],first);
 assert.equal(ids[handoffs[1]+1],second);
}
assert.equal(schema.questions.QID176.required,true);
console.log('Survey logic checks passed: A/B and B/A branches, repeated handoffs, active questions only.');
context.rawAnswers={
 'QID7':'5','QID7:5:text':'Self-described','QID176':'2',
 'QID1217253034:1':'4','QID185:3':'5','QID163:1':13,'QID186:1':12,
 'QID133:9':0,'QID192:9':17,'QID173':'Example notes',
 'QID193:1':'2','QID193:2':'4','QID193:3':'6','QID194:1':'7'
};
const exported=JSON.parse(vm.runInContext(
 'state={study_id:"P01",answers:rawAnswers}; JSON.stringify(snapshot())',
 context
));
assert.equal(exported.response_format_version,3);
assert.equal(exported.answers.participant.study_id,'P01');
assert.equal(exported.answers.participant.gender,'Other');
assert.equal(exported.answers.participant.gender_other,'Self-described');
assert.equal(exported.answers.scenario_order,'B_Cifar, then A_Pamap');
assert.equal(exported.answers.scenario_a.usability_agreement.meets_requirements,'Neither agree nor disagree');
assert.equal(exported.answers.scenario_b.usability_agreement.easy_to_use,'Somewhat agree');
assert.equal(exported.answers.scenario_a.workload_1_to_20.mental_demand,13);
assert.equal(exported.answers.scenario_b.workload_1_to_20.mental_demand,12);
assert.deepEqual(exported.answers.scenario_a.s_tias_1_to_7,{confidence:2,reliability:4,trust:6,mean:4});
assert.deepEqual(exported.answers.scenario_b.s_tias_1_to_7,{confidence:7});
assert.equal(exported.answers.scenario_a.tool_ratings_0_to_100.distracting,0);
assert.equal(exported.answers.scenario_b.tool_ratings_0_to_100.distracting,17);
assert.equal(exported.answers.experimenter_notes,'Example notes');
assert.equal(Object.hasOwn(exported,'answers_by_qid'),false);
console.log('Response export checks passed: readable labels, scenario groups, and named answers only.');
