const fs=require('node:fs');const vm=require('node:vm');const assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/web/app.js','utf8').split("$('begin').onclick")[0];
const context=vm.createContext({document:{getElementById:()=>({})},console,setTimeout,clearTimeout});vm.runInContext(source,context);
const schema=JSON.parse(fs.readFileSync(__dirname+'/web/survey.json','utf8'));
function route(order){context.inputSchema=schema;context.order=order;return JSON.parse(vm.runInContext('schema=inputSchema; state={answers:{QID176:order}}; JSON.stringify(makePages())',context));}
const a=route('1').flatMap(p=>p.ids),b=route('2').flatMap(p=>p.ids);
assert(a.indexOf('QID177')<a.indexOf('QID184'));
assert(b.indexOf('QID184')<b.indexOf('QID177'));
assert(a.includes('QID173')&&b.includes('QID173'));
assert(!a.includes('QID1')&&!b.includes('QID1'));
assert.equal(a.filter(id=>id==='QID175').length,3);
assert.equal(schema.questions.QID176.required,true);
console.log('Survey logic checks passed: A/B and B/A branches, repeated handoffs, active questions only.');
