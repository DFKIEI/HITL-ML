"""Export the active Qualtrics flow to a local, auditable survey schema."""
import json,re,html,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def plain(value):
    return html.unescape(re.sub('<[^>]+>', ' ', str(value or ''))).strip()
def convert(source=ROOT/'Latent_Space_VIS.qsf', output=ROOT/'study/web/survey.json'):
    raw=source.read_bytes(); document=json.loads(raw)
    elements=document['SurveyElements']
    flow=next(e['Payload']['Flow'] for e in elements if e['Element']=='FL')
    payload=next(e['Payload'] for e in elements if e['Element']=='BL')
    blocks=list(payload.values()) if isinstance(payload,dict) else payload
    active=set()
    def visit(nodes):
        for node in nodes:
            if node['Type'] in ('Block','Standard'):active.add(node['ID'])
            elif node['Type']=='Branch':visit(node.get('Flow',[]))
            elif node['Type']!='EndSurvey':raise ValueError(f"Unsupported flow node {node['Type']}")
    visit(flow)
    selected={b['ID']:dict(id=b['ID'],title=b.get('Description',''),elements=b.get('BlockElements',[])) for b in blocks if b['ID'] in active}
    ids={el['QuestionID'] for b in selected.values() for el in b['elements'] if el['Type']=='Question'}
    questions={}
    for e in elements:
        if e['Element']!='SQ' or e['Payload']['QuestionID'] not in ids:continue
        p=e['Payload'];qid=p['QuestionID']
        if p['QuestionType'] not in ('MC','TE','DB','Slider','Matrix'):raise ValueError(f'Unsupported {qid}')
        if p.get('DisplayLogic'):raise ValueError(f'Display logic needs explicit conversion: {qid}')
        def ordered(field,order):
            values=p.get(field) or {};keys=p.get(order) or list(values)
            return [dict(id=str(k),text=plain(values[str(k)]['Display']),textEntry=values[str(k)].get('TextEntry')=='true') for k in keys if str(k) in values]
        cfg=p.get('Configuration',{})
        questions[qid]=dict(id=qid,type=p['QuestionType'],selector=p.get('Selector'),text=plain(p.get('QuestionText')),
                            choices=ordered('Choices','ChoiceOrder'),answers=ordered('Answers','AnswerOrder'),
                            required=p.get('Validation',{}).get('Settings',{}).get('ForceResponse')=='ON' or qid=='QID176',
                            min=cfg.get('CSSliderMin',0),max=cfg.get('CSSliderMax',100),showValue=cfg.get('ShowValue',True),
                            labels=[plain(v.get('Display',v) if isinstance(v,dict) else v) for v in (p.get('Labels') or {}).values()])
    for b in selected.values():
        b['elements']=[dict(type=el['Type'],question=el.get('QuestionID')) for el in b['elements'] if el['Type'] in ('Question','Page Break')]
    schema=dict(title='Latent Space VIS',source=source.name,source_sha256=hashlib.sha256(raw).hexdigest(),
                flow=flow,blocks=selected,questions=questions)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(schema,indent=2,ensure_ascii=False))
    return schema
if __name__=='__main__':
    schema=convert();print(f"Exported {len(schema['questions'])} active questions in {len(schema['blocks'])} blocks.")
