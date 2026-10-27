const path=require('path');
const puppeteer=require(path.join(process.env.HOME,'.agents/skills/deck-kit/node_modules/puppeteer-core'));
(async()=>{const [html,out]=process.argv.slice(2);
const b=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:'new',args:['--no-sandbox']});
const p=await b.newPage(); await p.setViewport({width:1700,height:1200,deviceScaleFactor:2});
await p.goto('file://'+path.resolve(html),{waitUntil:'networkidle0'}); await p.evaluateHandle('document.fonts.ready');
const el=await p.$('#cv'); await el.screenshot({path:out}); await b.close(); console.log(out);})().catch(e=>{console.error(e);process.exit(1)});
