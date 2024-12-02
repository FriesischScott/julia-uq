import{_ as i,c as a,a5 as e,o as t}from"./chunks/framework.fh0vx_DB.js";const k=JSON.parse('{"title":"High performance computing","description":"","frontmatter":{},"headers":[],"relativePath":"manual/hpc.md","filePath":"manual/hpc.md","lastUpdated":null}'),n={name:"manual/hpc.md"};function l(o,s,h,r,p,d){return t(),a("div",null,s[0]||(s[0]=[e(`<h1 id="High-performance-computing" tabindex="-1">High performance computing <a class="header-anchor" href="#High-performance-computing" aria-label="Permalink to &quot;High performance computing {#High-performance-computing}&quot;">​</a></h1><h2 id="Slurm-job-arrays" tabindex="-1">Slurm job arrays <a class="header-anchor" href="#Slurm-job-arrays" aria-label="Permalink to &quot;Slurm job arrays {#Slurm-job-arrays}&quot;">​</a></h2><p>When sampling large simulation models, or complicated workflows, Julia&#39;s inbuilt parallelism is sometimes insufficient. Job arrays are a useful feature of the slurm scheduler which allow you to run many similar jobs, which differ by an index (for example a sample number). This allows <code>UncertaintyQuantification.jl</code> to run heavier simulations (for example, simulations requiring multiple nodes), by offloading model sampling to an HPC machine using slurm. This way, <code>UncertaintyQuantification.jl</code> can be started on a single worker, and the HPC machine handles the rest.</p><p>For more information on job arrays, see: <a href="https://slurm.schedmd.com/job_array.html" target="_blank" rel="noreferrer">job arrays</a>.</p><h2 id="slurminterface" tabindex="-1">SlurmInterface <a class="header-anchor" href="#slurminterface" aria-label="Permalink to &quot;SlurmInterface&quot;">​</a></h2><p>When <code>SlurmInterface</code> is passed to an <code>ExternalModel</code>, a slurm job array script is automatically generated and executed. Julia waits for this job to finish before extracting results and proceeding.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">options </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dict</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;account&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;HPC_account_1&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;partition&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CPU_partition&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;job-name&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;UQ_array&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;nodes&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;1&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;ntasks&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;32&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;time&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;01:00:00&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">slurm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SlurmInterface</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    options;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    throttle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    extras</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;load python3&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Here <code>account</code> is your account (provided by your HPC admin/PI), and <code>partition</code> specifies the queue that jobs will be submitted to (ask admin if unsure). <code>nodes</code> and <code>ntasks</code> are the number of nodes and CPUs that your individual simulations requires. Depending on your HPC machine, each node has a specific number of CPUs. If your application requires more CPUs than are available per node, you can use multiple nodes. Through <code>options</code> the <code>SlurmInterface</code> supports all options of <code>SBATCH</code> except for <code>array</code> since the job array is constructed dynamically.</p><p>The parameter <code>time</code> specifies the maximum time that each simulation will be run for, before being killed.</p><div class="warning custom-block"><p class="custom-block-title">Individual model runs VS overall batch</p><p><code>nodes</code>, <code>ntasks</code>, and <code>time</code> are parameters required for each <em>individual</em> model evaluation, not the entire batch. For example, if you are running a large FEM simulation that requires 100 CPUs to evaluate one sample, and your HPC machine has 50 CPUs per node, you would specify <code>nodes = 2</code> and <code>ntasks = 100</code>.</p></div><div class="tip custom-block"><p class="custom-block-title">Compiling with MPI</p><p>If your model requires multiple <code>nodes</code>, it may be best to compile your application with MPI, if your model allows for it. Please check your application&#39;s documentation for compiling with MPI.</p></div><p>Any commands in <code>extras</code> will be executed before you model is run, for example loading any module files or data your model requires. Multiple commands can be passed: <code>extras = [&quot;load python&quot;, &quot;python3 get_data.py&quot;]</code>.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>If your <code>extras</code> command requires <code>&quot;&quot;</code> or <code>$</code> symbols, they must be properly escaped as <code>\\&quot;\\&quot;</code> and <code>\\$</code>.</p></div><p>The job array task throttle, which is the number of samples that will be run concurrently at any given time, is specified by <code>throttle</code>. For example, when running a <code>MonteCarlo</code> simulation with 2000 samples, and <code>throttle = 50</code>, 2000 model evaluations will be run in total, but only 50 at the same time. If left empty, your scheduler&#39;s default throttle will be used. Sometimes the scheduler limits the maximum size of a single job array. In these cases, the maximum size can be set through the <code>batchsize</code> parameter. This will separate the jobs into multiple smaller arrays.</p><h2 id="Testing-your-HPC-configuration" tabindex="-1">Testing your HPC configuration <a class="header-anchor" href="#Testing-your-HPC-configuration" aria-label="Permalink to &quot;Testing your HPC configuration {#Testing-your-HPC-configuration}&quot;">​</a></h2><p>Slurm is tested <em>only</em> on linux systems, not Mac or Windows. When testing <code>UncertaintyQuantification.jl</code> locally, we use a dummy function <code>test/test_utilities/sbatch</code> to mimic an HPC scheduler.</p><div class="warning custom-block"><p class="custom-block-title">Testing locally on Linux</p><p>Certain Slurm tests may fail unless <code>test/test_utilities/</code> is added to PATH. To do this: <code>export PATH=UncertaintyQuantification.jl/test/test_utilities/:$PATH</code>. Additionally, <em>actual</em> slurm submissions may fail if <code>test/test_utilities/sbatch</code> is called in place of your system installation. To find out which sbatch you&#39;re using, call <code>which sbatch</code>.</p></div><p>If you&#39;d like to <strong>actually</strong> test the Slurm interface your HPC machine:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;UncertaintyQuantification&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; test_args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;HPC&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;YOUR_ACCOUNT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;YOUR_PARTITION&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p>or if you have a local clone, from the top directory:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">--</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">project</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; test_args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;HPC&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;YOUR_ACCOUNT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;YOUR_PARTITION&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p><code>YOUR_ACCOUNT</code> and <code>YOUR_PARTITION</code> should be replaced with your account and partition you wish to use for testing. This test will submit 4 slurm job arrays, of a lightweight calculation (&gt; 1 minute per job) requiring 1 core/task each.</p><h3 id="usage" tabindex="-1">Usage <a class="header-anchor" href="#usage" aria-label="Permalink to &quot;Usage&quot;">​</a></h3><p>See <a href="./examples/hpc">examples/HPC</a> for a detailed example.</p>`,24)]))}const u=i(n,[["render",l]]);export{k as __pageData,u as default};
