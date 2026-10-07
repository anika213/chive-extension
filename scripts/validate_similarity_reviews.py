"""Validate review links/data and optionally exercise the local interactive HTML."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import os
import signal
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent.parent/'results/petri_latest_qualified/similarity'


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--browser',action='store_true');args=p.parse_args()
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='a' and a.get('href'):self.targets.append(a['href'])
            if tag=='img' and a.get('src'):self.targets.append(a['src'])
    total=0
    for cluster in range(6):
        page=ROOT/f'cluster_{cluster}.html';text=page.read_text()
        parser=Links();parser.feed(text)
        for link in parser.targets:assert (ROOT/link.split('#')[0]).exists(),link
        payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S)[1])
        assert payload==json.loads((ROOT/f'cluster_{cluster}/data.json').read_text())
        assert payload['n']==len(payload['members'])
        total+=payload['n']
        for r in payload['members']:
            target,anchor=r['cause_review_url'].split('#')
            assert (ROOT/target).exists() and f'id="{anchor}"' in (ROOT/target).read_text()
        for k,labels in payload['cuts'].items():assert len(set(labels))==int(k)
    assert total==172
    subprocess.run(['node','--check',str(ROOT/'interaction.js')],check=True)
    print('Passed: all six pages, 172 members, JSON payloads, cause links, hierarchy cuts, and JavaScript syntax.')
    if not args.browser:return
    check=ROOT/'_interaction_check.html'
    js=r'''<script>
try {
 const assert=(v,m)=>{if(!v)throw new Error(m)};
 assert(document.querySelectorAll('.member').length===37,'initial members');
 document.getElementById('cut').value='4';document.getElementById('cut').dispatchEvent(new Event('change'));
 assert(document.querySelectorAll('.group').length===4,'cut group count');
 assert(document.querySelectorAll('.member').length===37,'cut member count');
 document.getElementById('sort').value='central';document.getElementById('sort').dispatchEvent(new Event('change'));
 for(const group of document.querySelectorAll('.group')){const ranks=[...group.querySelectorAll('.member h4')].map(x=>Number(x.textContent.match(/centrality rank (\d+)/)[1]));assert(ranks.every((r,i)=>i===0||r>=ranks[i-1]),'centrality order');}
 const rects=document.querySelectorAll('#heatmap svg rect');assert(rects.length===37*37+4,'matrix and subgroup borders');
 rects[1].dispatchEvent(new MouseEvent('click',{bubbles:true}));assert(document.querySelector('#pair h3').textContent.includes('cosine'),'pair click');
 document.getElementById('reset').click();assert(document.querySelectorAll('.group').length===1,'reset');
 assert(document.querySelectorAll('#pairs tr').length===15,'closest pairs');
 document.body.dataset.qa='passed';
} catch(e){document.body.dataset.qa='failed';document.body.dataset.qaError=e.message;}
</script>'''
    check.write_text((ROOT/'cluster_4.html').read_text().replace('</body>',js+'</body>'))
    profile=tempfile.mkdtemp(prefix='chive-similarity-chrome-')
    cmd=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
         '--disable-background-networking','--disable-component-update','--disable-sync','--disable-crash-reporter','--user-data-dir='+profile,
         '--window-size=1400,1200','--virtual-time-budget=2000','--screenshot='+str(ROOT/'interaction_preview.png'),'--dump-dom',check.as_uri()]
    try:
        # Chrome on macOS can keep helper processes alive after producing its DOM.
        # Capture to files, then terminate only this isolated browser process group.
        with tempfile.TemporaryFile(mode='w+') as stdout, tempfile.TemporaryFile(mode='w+') as stderr:
            process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=5)
            stdout.seek(0);dom=stdout.read();stderr.seek(0);errors=stderr.read()
        match=re.search(r'<body[^>]*data-qa="([^"]+)"[^>]*>',dom)
        print('BROWSER_EXIT',process.returncode,'QA',match.group(0) if match else 'No result')
        assert match and match[1]=='passed','Browser interaction validation failed: '+errors[-1000:]
        (ROOT/'browser_validation.json').write_text(json.dumps({'status':'passed','cluster':4,'members':37,'cut':4,
            'centrality_sort':True,'heatmap_pair_click':True,'reset':True,'closest_pairs':15},indent=2)+'\n')
    finally:check.unlink(missing_ok=True)


if __name__=='__main__':main()
