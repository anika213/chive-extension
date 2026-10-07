"""Exercise the single-investigation example in isolated local headless Chrome."""
from pathlib import Path
from html.parser import HTMLParser
import json
import os
import re
import signal
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    original = ROOT / 'example.html'
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links=[]; self.ids=set()
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if a.get('id'): self.ids.add(a['id'])
            if tag=='a' and a.get('href'): self.links.append(a['href'])
    parser=Links();parser.feed(original.read_text())
    for link in parser.links:
        if link.startswith('#'): assert link[1:] in parser.ids
        else: assert (ROOT/link).exists(),link
    subprocess.run(['node','--check',str(ROOT/'examples/investigation_parts_example.js')],check=True)
    qa=r'''<script>
try{
 const assert=(x,m)=>{if(!x)throw new Error(m)};
 assert(D.experiments.length===13,'13 experiments');
 assert(document.getElementById('experiment').options.length===13,'experiment selector');
 assert(document.getElementById('experiment').value==='12','default exp12');
 assert(document.getElementById('effect-headline').textContent.includes('21/30 → 2/30'),'correct initial rates');
 assert(document.getElementById('baseline-response').options.length===30,'30 baseline samples');
 assert(document.getElementById('counter-response').options.length===30,'30 experimental samples');
 assert(document.getElementById('baseline-label').textContent==='Classifier: YES','baseline sample chosen yes');
 assert(document.getElementById('counter-label').textContent==='Classifier: NO','counter sample chosen no');
 assert(document.querySelectorAll('#effects-table tr').length===13,'all effect rows');
 assert(document.getElementById('summary-cleaned').textContent===current().preprocessed.cause_text,'exact clean text');
 for(const e of D.experiments){
  document.getElementById('experiment').value=String(e.exp_num);document.getElementById('experiment').dispatchEvent(new Event('change'));
  assert(document.getElementById('counter-response').options.length===e.completions.length,'sample count');
  assert(document.getElementById('edited-context').children.length===e.messages.length,'edited context');
  assert(document.getElementById('diff-before').textContent===D.messages[D.messages.length-1].content,'exact baseline diff');
  assert(document.getElementById('diff-after').textContent===e.messages[e.messages.length-1].content,'exact experimental diff');
 }
 document.querySelectorAll('#effects-table button')[4].click();assert(current().exp_num===5,'table experiment navigation');
 document.getElementById('experiment').value='12';document.getElementById('experiment').dispatchEvent(new Event('change'));
 document.getElementById('counter-response').value='29';document.getElementById('counter-response').dispatchEvent(new Event('change'));
 assert(document.getElementById('counter-output').textContent===current().completions[29],'response selection');
 renderExperiment();document.documentElement.style.scrollBehavior='auto';
 window.scrollTo({top:0,behavior:'instant'});
 setTimeout(()=>window.scrollTo({top:0,behavior:'instant'}),250);
 document.body.dataset.qa='passed';
}catch(e){document.body.dataset.qa='failed';document.body.dataset.qaError=e.message}
</script>'''
    check=ROOT/'_parts_example_check.html'
    check.write_text(original.read_text().replace('</body>',qa+'</body>'))
    profile=tempfile.mkdtemp(prefix='chive-parts-example-chrome-')
    cmd=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
         '--disable-background-networking','--disable-component-update','--disable-sync','--disable-crash-reporter','--user-data-dir='+profile,
         '--window-size=1440,1500','--run-all-compositor-stages-before-draw','--virtual-time-budget=2000','--screenshot='+str(ROOT/'examples/investigation_parts_example_preview.png'),
         '--dump-dom',check.as_uri()]
    try:
        with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
            stdout.seek(0);dom=stdout.read();stderr.seek(0);error=stderr.read()
        tag=re.search(r'<body[^>]*data-qa="([^"]+)"[^>]*>',dom)
        assert tag and tag[1]=='passed', (tag.group(0) if tag else error[-1000:])
        result={'status':'passed','all_13_experiment_selectors_checked':True,'exact_context_diff_texts_preserved':True,
                'response_selector_checked':True,'baseline_yes_and_counter_no_initial_selection':True,'table_navigation_checked':True,
                'all_static_links_exist':True,'javascript_syntax_checked':True}
        (ROOT/'examples/investigation_parts_example_browser_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    finally:check.unlink(missing_ok=True)


if __name__=='__main__':main()
