import{_ as n,c as o,a5 as s,j as e,a as t,G as l,B as d,o as r}from"./chunks/framework.Ios9mXB1.js";const v=JSON.parse('{"title":"Models","description":"","frontmatter":{},"headers":[],"relativePath":"api/models.md","filePath":"api/models.md","lastUpdated":null}'),p={name:"api/models.md"},c={class:"jldocstring custom-block",open:""},h={class:"jldocstring custom-block",open:""},u={class:"jldocstring custom-block",open:""},k={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""};function y(m,a,b,g,F,E){const i=d("Badge");return r(),o("div",null,[a[16]||(a[16]=s('<h1 id="models" tabindex="-1">Models <a class="header-anchor" href="#models" aria-label="Permalink to &quot;Models&quot;">​</a></h1><h2 id="index" tabindex="-1">Index <a class="header-anchor" href="#index" aria-label="Permalink to &quot;Index&quot;">​</a></h2><ul><li><a href="#UncertaintyQuantification.Model"><code>UncertaintyQuantification.Model</code></a></li><li><a href="#UncertaintyQuantification.ParallelModel"><code>UncertaintyQuantification.ParallelModel</code></a></li><li><a href="#UncertaintyQuantification.UQModel"><code>UncertaintyQuantification.UQModel</code></a></li><li><a href="#UncertaintyQuantification.evaluate!-Tuple{ParallelModel, DataFrame}"><code>UncertaintyQuantification.evaluate!</code></a></li><li><a href="#UncertaintyQuantification.evaluate!-Tuple{Model, DataFrame}"><code>UncertaintyQuantification.evaluate!</code></a></li></ul><h2 id="types" tabindex="-1">Types <a class="header-anchor" href="#types" aria-label="Permalink to &quot;Types&quot;">​</a></h2>',4)),e("details",c,[e("summary",null,[a[0]||(a[0]=e("a",{id:"UncertaintyQuantification.UQModel",href:"#UncertaintyQuantification.UQModel"},[e("span",{class:"jlbinding"},"UncertaintyQuantification.UQModel")],-1)),a[1]||(a[1]=t()),l(i,{type:"info",class:"jlObjectType jlType",text:"Type"})]),a[2]||(a[2]=e("p",null,"Abstract supertype for all model types",-1)),a[3]||(a[3]=e("p",null,[e("a",{href:"https://github.com/FriesischScott/UncertaintyQuantification.jl/blob/7acd79737e70656b1102759b41adb99d1433f8f1/src/UncertaintyQuantification.jl#L34-L36",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",h,[e("summary",null,[a[4]||(a[4]=e("a",{id:"UncertaintyQuantification.Model",href:"#UncertaintyQuantification.Model"},[e("span",{class:"jlbinding"},"UncertaintyQuantification.Model")],-1)),a[5]||(a[5]=t()),l(i,{type:"info",class:"jlObjectType jlType",text:"Type"})]),a[6]||(a[6]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>The function <code>f</code> must accept a <code>DataFrame</code> and return the result of the model for each row in the <code>DataFrame</code> as a vector. The <code>name</code> is used to add the output to the <code>DataFrame</code>.</p><p><a href="https://github.com/FriesischScott/UncertaintyQuantification.jl/blob/7acd79737e70656b1102759b41adb99d1433f8f1/src/models/model.jl#L1-L6" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",u,[e("summary",null,[a[7]||(a[7]=e("a",{id:"UncertaintyQuantification.ParallelModel",href:"#UncertaintyQuantification.ParallelModel"},[e("span",{class:"jlbinding"},"UncertaintyQuantification.ParallelModel")],-1)),a[8]||(a[8]=t()),l(i,{type:"info",class:"jlObjectType jlType",text:"Type"})]),a[9]||(a[9]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ParallelModel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>The <code>ParallelModel</code> does what the <code>Model</code> does with a small difference. The function <code>f</code> is passed a <code>DataFrameRow</code> not the full <code>DataFrame</code>. If workers (through <code>Distributed</code>) are present, the rows are evaluated in parallel.</p><p><a href="https://github.com/FriesischScott/UncertaintyQuantification.jl/blob/7acd79737e70656b1102759b41adb99d1433f8f1/src/models/model.jl#L12-L18" target="_blank" rel="noreferrer">source</a></p>',3))]),a[17]||(a[17]=e("h2",{id:"methods",tabindex:"-1"},[t("Methods "),e("a",{class:"header-anchor",href:"#methods","aria-label":'Permalink to "Methods"'},"​")],-1)),e("details",k,[e("summary",null,[a[10]||(a[10]=e("a",{id:"UncertaintyQuantification.evaluate!-Tuple{Model, DataFrame}",href:"#UncertaintyQuantification.evaluate!-Tuple{Model, DataFrame}"},[e("span",{class:"jlbinding"},"UncertaintyQuantification.evaluate!")],-1)),a[11]||(a[11]=t()),l(i,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[12]||(a[12]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">evaluate!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(m</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, df</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DataFrame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Calls <code>m.func</code> with <code>df</code> and adds the result to the <code>DataFrame</code> as a column <code>m.name</code></p><p><a href="https://github.com/FriesischScott/UncertaintyQuantification.jl/blob/7acd79737e70656b1102759b41adb99d1433f8f1/src/models/model.jl#L32-L36" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",f,[e("summary",null,[a[13]||(a[13]=e("a",{id:"UncertaintyQuantification.evaluate!-Tuple{ParallelModel, DataFrame}",href:"#UncertaintyQuantification.evaluate!-Tuple{ParallelModel, DataFrame}"},[e("span",{class:"jlbinding"},"UncertaintyQuantification.evaluate!")],-1)),a[14]||(a[14]=t()),l(i,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[15]||(a[15]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">evaluate!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(m</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ParallelModel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, df</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DataFrame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Calls <code>m.func</code> for each row of <code>df</code> and adds the result to the <code>DataFrame</code> as a column <code>m.name</code>. If workers are added through <code>Distributed</code>, the rows will be evaluated in parallel.</p><p><a href="https://github.com/FriesischScott/UncertaintyQuantification.jl/blob/7acd79737e70656b1102759b41adb99d1433f8f1/src/models/model.jl#L42-L47" target="_blank" rel="noreferrer">source</a></p>',3))])])}const U=n(p,[["render",y]]);export{v as __pageData,U as default};
