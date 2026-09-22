"""Compile the packaged current TeX sources, without regenerating science."""
import argparse, json, shutil, subprocess
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();(source/'build').mkdir(exist_ok=True)
    engine=shutil.which('xelatex') or '/Library/TeX/texbin/xelatex'
    bibtex=shutil.which('bibtex') or '/Library/TeX/texbin/bibtex'
    logs=[]
    for n in ('supplementary','main_paper4','bibtex','main_paper4','main_paper4','supplementary','supplementary','main_paper4'):
        cmd=[bibtex,'build/main_paper4'] if n=='bibtex' else [engine,'-interaction=nonstopmode','-halt-on-error','-output-directory=build',n+'.tex']
        r=subprocess.run(cmd,cwd=source,capture_output=True,text=True);logs.append(r.stdout+r.stderr)
        a.log.parent.mkdir(parents=True,exist_ok=True);a.log.write_text('\n'.join(logs))
        if r.returncode:raise RuntimeError(logs[-1][-3000:])
    warnings={}
    for n in ('main_paper4','supplementary'):
        lines=(source/'build'/f'{n}.log').read_text().splitlines()
        warnings[n]=[x for x in lines if any(t in x for t in ('undefined','Overfull','Missing character','Float too large'))]
        assert not warnings[n],warnings
    print(json.dumps(dict(source=str(source),status='passed',warnings=warnings),indent=2))
