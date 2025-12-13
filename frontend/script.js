// ===== API BASE (local only) =====
const API_BASE = "http://127.0.0.1:8000";

// ===== AUTH HELPERS =====
function token() { return localStorage.getItem("token") || ""; }
function authHeaders() { const t = token(); return t ? { Authorization: `Bearer ${t}` } : {}; }
function ensureLoggedIn(){ if(!token()) location.href = "login.html"; }
ensureLoggedIn();

// ===== ELEMENTS =====
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const logoutBtn = document.getElementById("logoutBtn");
const themeBtn = document.getElementById("themeBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const sidebar = document.getElementById("sidebar");
const chatList = document.getElementById("chatList");
const newChatBtn = document.getElementById("newChatBtn");
const toggleSidebarBtn = document.getElementById("toggleSidebar");

// media / files
const uploadBtn = document.getElementById("uploadBtn");
const genBtn = document.getElementById("genBtn");
const imgInput = document.getElementById("imgInput");
const pdfBtn = document.getElementById("pdfBtn");
const pdfInput = document.getElementById("pdfInput");

// NEW: ML features
const searchBtn = document.getElementById("searchBtn");
const routeBtn = document.getElementById("routeBtn");
const nextBtn = document.getElementById("nextBtn");

// ===== MARKDOWN / HIGHLIGHT =====
if (window.marked) {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: (code, lang) => {
      try { return hljs.highlightAuto(code, lang ? [lang] : undefined).value; }
      catch { return code; }
    }
  });
}

// ===== STATE (server-backed) =====
let chats = [];    // [{id,title,updated_at}]
let activeId = ""; // current chat id

// ===== API WRAPPER =====
async function api(path, options = {}) {
  const opts = {
    ...options,
    headers: {
      ...(options.headers || {}),
      "Content-Type": "application/json",
      ...authHeaders(),
    }
  };
  // Don't set content-type for FormData
  if (options.body instanceof FormData) {
    delete opts.headers["Content-Type"];
  }
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (res && res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    location.href = "login.html";
    return;
  }
  return res.json();
}

// ===== THEME =====
(function initTheme(){
  const saved = localStorage.getItem("theme") || "dark";
  document.body.setAttribute("data-theme", saved);
})();
themeBtn?.addEventListener("click", () => {
  const next = document.body.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

// ===== LOGOUT =====
logoutBtn?.addEventListener("click", ()=>{
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  location.href = "login.html";
});

// ===== HEALTH PING =====
async function ping(){
  setStatus("idle");
  try{
    const res = await fetch(`${API_BASE}/`);
    if(res.ok){ setStatus("online"); statusText.textContent = "Online"; }
    else { setStatus("offline"); statusText.textContent = "Offline"; }
  }catch{ setStatus("offline"); statusText.textContent = "Offline"; }
}
function setStatus(state){
  statusDot.classList.remove("online","offline","idle");
  statusDot.classList.add(state);
}
ping(); setInterval(ping, 15000);

// ===== TEXTAREA =====
function autosize(){
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize); autosize();
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ===== SIDEBAR =====
toggleSidebarBtn?.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});
newChatBtn?.addEventListener("click", () => newChat());

function renderSidebar(){
  chatList.innerHTML = "";
  if (!chats || chats.length === 0) return;
  chats.forEach(c => {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === activeId ? " active" : "");
    item.innerHTML = `
      <div class="logo">💬</div>
      <div class="title">${(c.title || "New chat")}</div>
      <div class="time">${c.updated_at ? new Date(c.updated_at*1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) : ""}</div>
      <div class="trash" title="Delete">🗑</div>
    `;
    item.addEventListener("click", (e)=>{
      if (e.target && e.target.classList.contains("trash")) return;
      switchChat(c.id);
    });
    item.querySelector(".trash").addEventListener("click", async (e)=>{
      e.stopPropagation();
      await api(`/api/chats/${c.id}`, { method: "DELETE" });
      await loadChats();
    });
    chatList.appendChild(item);
  });
}

// ===== MESSAGE UI =====
let typingEl = null;
function messageEl(role, timeText){
  const msg = document.createElement("div");
  msg.className = "msg" + (role === "user" ? " right" : "");
  const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = role === "user" ? "🧑" : "🤖";
  const bubble = document.createElement("div"); bubble.className = "bubble";
  const content = document.createElement("div"); content.className = "content";
  const meta = document.createElement("div"); meta.className = "meta";
  const time = document.createElement("span"); time.className = "time"; time.textContent = timeText || new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  meta.appendChild(time);
  if (role !== "user"){
    const copy = document.createElement("span"); copy.className = "copy"; copy.title = "Copy response"; copy.textContent = "Copy";
    copy.addEventListener("click", ()=>{ navigator.clipboard.writeText(content.innerText); copy.textContent="Copied!"; setTimeout(()=>copy.textContent="Copy", 1100); });
    meta.appendChild(copy);
  }
  bubble.appendChild(content); bubble.appendChild(meta);
  msg.appendChild(avatar); msg.appendChild(bubble);
  return { msg, content };
}

function addMessageEl(role, text){
  const { msg, content } = messageEl(role);
  if (window.marked) content.innerHTML = marked.parse(text || "");
  else content.textContent = text || "";
  chat.appendChild(msg);
  chat.querySelectorAll("pre code").forEach(block => window.hljs && hljs.highlightElement(block));
  chat.scrollTop = chat.scrollHeight;
}
function showTyping(){
  if (typingEl) return;
  const { msg, content } = messageEl("assistant");
  content.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
  typingEl = msg; chat.appendChild(typingEl); chat.scrollTop = chat.scrollHeight;
}
function hideTyping(){ if (typingEl){ typingEl.remove(); typingEl = null; } }

// ===== LOAD & RENDER CHAT =====
async function loadChats(){
  const data = await api("/api/chats");
  chats = (data && data.chats) ? data.chats : [];
  if(!activeId && chats.length) activeId = chats[0].id;
  renderSidebar();
  await renderChat();
}

async function renderChat(){
  chat.innerHTML = "";
  if(!activeId){
    addMessageEl("assistant", "Hey! I'm ready. Create a new chat or select one from the left.");
    return;
  }
  const data = await api(`/api/chats/${activeId}`);
  const msgs = (data && data.messages) ? data.messages : [];
  if(msgs.length === 0){
    addMessageEl("assistant", "This chat is empty. Say something!");
  } else {
    for(const m of msgs){
      addMessageEl(m.role === "assistant" ? "assistant" : "user", m.content);
    }
  }
}

async function newChat(){
  const res = await api("/api/chats/new", { method: "POST" });
  activeId = res.chat_id;
  await api("/api/reset", { method: "POST", body: JSON.stringify({ chat_id: activeId }) });
  await loadChats();
  input.focus();
}

async function switchChat(id){
  activeId = id;
  renderSidebar();
  await renderChat();
  sidebar.classList.remove("open");
}

// ========= RAG helper (client-side context injection) =========
async function tryRagAugment(question, k = 6) {
  try {
    const search = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query: question, chat_id: activeId || "default", k })
    });

    if (!search || !Array.isArray(search.results) || !search.results.length) {
      return null; // no indexed docs; skip RAG
    }

    const top = search.results.slice(0, Math.min(k, search.results.length));
    const ctx = top.map((r, i) => {
      const src = r.source ? ` (source: ${r.source}${typeof r.section_idx==="number" ? `, sec ${r.section_idx}` : ""})` : "";
      return `[${i+1}]${src}\n${(r.text||"").trim()}`;
    }).join("\n\n");

    const instr =
      "Use ONLY the context blocks [1..n] below to answer the user's question.\n" +
      "Be concise, and add bracket citations like [1], [2] after facts you pull from a block.\n" +
      "If the context is insufficient, ask a brief clarifying question.";

    const augmented = `${instr}\n\nContext:\n${ctx}\n\nQuestion: ${question}`;

    const sources = top.map(r => (r.source || "").trim()).filter(Boolean);
    return { augmented, sources };
  } catch {
    return null;
  }
}

// ===== SEND FLOW (auto RAG + feedback) =====
sendBtn?.addEventListener("click", sendMessage);
// NOTE: we already have the Enter-to-send handler above under TEXTAREA; don't add another here.

async function sendMessage(){
  const text = input.value.trim();
  if(!text) return;
  if(!activeId){
    const res = await api("/api/chats/new", { method: "POST" });
    activeId = res.chat_id;
  }

  addMessageEl("user", text);
  input.value = ""; autosize();

  showTyping();
  try{
    // Try to fetch top reranked snippets and inject as context
    let rag = await tryRagAugment(text, 6);

    const payload = { user_input: rag ? rag.augmented : text, chat_id: activeId };
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    hideTyping();
    addMessageEl("assistant", (data && data.response) || "(no response)");

    // Prefer server-declared sources_used (from answerability auto-wire), else fallback to RAG sources
    const usedSources =
      (data && Array.isArray(data.sources_used) && data.sources_used.length ? data.sources_used : (rag && rag.sources) || []);

    if (usedSources.length) {
      renderSourceFeedback(usedSources, "Sources used");
    }

    // refresh sidebar meta (titles/timestamps) without clearing current chat UI
    const chatsData = await api("/api/chats");
    chats = (chatsData && chatsData.chats) ? chatsData.chats : chats;
    renderSidebar();

  }catch(e){
    hideTyping();
    addMessageEl("assistant", "⚠️ Network error. Please try again.");
  }
}

// Clear current conversation (server + memory)
clearBtn?.addEventListener("click", async () => {
  if(!activeId) return;
  await api(`/api/chats/${activeId}/clear`, { method: "POST" });
  await api("/api/reset", { method: "POST", body: JSON.stringify({ chat_id: activeId }) });
  await renderChat();
});

// ===== IMAGE: Analyze (Vision) =====
uploadBtn?.addEventListener("click", () => imgInput.click());
imgInput?.addEventListener("change", async () => {
  const file = imgInput.files && imgInput.files[0];
  if (!file) return;

  const prompt = window.prompt(
    "Describe what you want me to do with this image:",
    "Describe this image"
  );

  const url = URL.createObjectURL(file);
  addMessageEl("user", `(uploaded image)\n\n![image](${url})`);
  showTyping();

  try {
    const form = new FormData();
    form.append("image", file);
    form.append("prompt", prompt || "Describe this image");
    form.append("chat_id", activeId || "default");

    const res = await fetch(`${API_BASE}/api/vision`, {
      method: "POST",
      headers: { ...authHeaders() },
      body: form,
    });

    const data = await res.json();
    hideTyping();
    addMessageEl("assistant", data.response || "(no response)");
    await loadChats();
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ Image analyze failed.");
  } finally {
    imgInput.value = "";
  }
});

// ===== IMAGE: Generate (HF forced; no reload after) =====
genBtn?.addEventListener("click", async () => {
  const prompt = window.prompt(
    "Image prompt (what to create)?",
    "A cute corgi wearing sunglasses, studio lighting"
  );
  if (!prompt) return;

  addMessageEl("user", `Generate: ${prompt}`);
  showTyping();

  try {
    const res = await fetch(`${API_BASE}/api/image/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        prompt,
        aspect_ratio: "1:1",
        count: 1,
        provider: "hf",                 // force HF to match backend behavior
        chat_id: activeId || "default"
      }),
    });

    const data = await res.json();
    hideTyping();

    if (!res.ok) {
      addMessageEl("assistant", `⚠️ Image generation failed: ${data?.detail || res.statusText}`);
      return;
    }

    if (data.images && data.images.length) {
      data.images.forEach((src) => addMessageEl("assistant", `![generated](${src})`));
      // IMPORTANT: do NOT call loadChats() here; generated images are not persisted server-side.
    } else {
      addMessageEl("assistant", data?.detail || "No image returned.");
    }
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ Network error while generating image.");
  }
});

// ===== PDF: Read (Summarize or QA) =====
pdfBtn?.addEventListener("click", () => pdfInput.click());
pdfInput?.addEventListener("change", async () => {
  const file = pdfInput.files && pdfInput.files[0];
  if (!file) return;

  const mode = window.prompt("Type 'q' to ask a question about the PDF, or press OK to Summarize.", "");
  let question = null;
  if (mode && mode.trim().toLowerCase() === "q") {
    question = window.prompt("Ask your question about the PDF:", "What are the key takeaways?");
  }

  addMessageEl("user", `(uploaded PDF) ${question ? "Q&A" : "Summarize"} — ${file.name}`);
  showTyping();

  try {
    const form = new FormData();
    form.append("pdf", file);
    form.append("chat_id", activeId || "default");
    if (question) form.append("question", question);

    const res = await fetch(`${API_BASE}/api/pdf/read`, {
      method: "POST",
      headers: { ...authHeaders() },
      body: form,
    });
    const data = await res.json();
    hideTyping();

    if (data.response) {
      addMessageEl("assistant", data.response);
      await loadChats();
    } else {
      addMessageEl("assistant", data?.detail || "Could not process the PDF.");
    }
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ PDF processing failed.");
  } finally {
    pdfInput.value = "";
  }
});

// ===== NEW: Personalized Retrieval & Reranking (semantic search) =====
searchBtn?.addEventListener("click", async () => {
  const q = window.prompt("Search your uploaded docs / context:", "quarterly revenue growth");
  if (!q) return;

  addMessageEl("user", `Search: ${q}`);
  showTyping();

  try{
    const data = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query: q, chat_id: activeId || "default", k: 8 })
    });
    hideTyping();

    if (!data || !data.results) {
      addMessageEl("assistant", data?.detail || "No results.");
      return;
    }

    const lines = data.results.map((r, i) => {
      const snip = (r.text || "").slice(0, 280).replace(/\n/g,' ');
      const src = r.source ? ` _(source: ${r.source}, sec ${r.section_idx ?? "-"})_` : "";
      const sc = (typeof r.final === "number" ? r.final : r.score);
      const scStr = (typeof sc === "number" && sc.toFixed) ? sc.toFixed(3) : String(sc ?? "");
      return `${i+1}. **score:** ${scStr}${src}\n> ${snip}${snip.length>=280?"…":""}`;
    }).join("\n\n");

    addMessageEl("assistant", `**Top results** (personalized & reranked):\n\n${lines}`);

    // Offer source feedback buttons based on these results
    renderSourceFeedback(data.results.map(r => r.source).filter(Boolean), "Relevant sources");

  }catch(e){
    hideTyping();
    addMessageEl("assistant", "⚠️ Search failed.");
  }
});

// ===== NEW: Learned Intent Router (fixed) =====
routeBtn?.addEventListener("click", async () => {
  // use selection or an input
  let text = (window.getSelection?.().toString() || "").trim();
  if (!text) {
    text = window.prompt(
      "Enter a request to route (Chat / PDF-QA / Vision / Image):",
      "Summarize the attached PDF"
    );
    if (!text) return;
  }

  addMessageEl("user", `Route this: "${text}"`);
  showTyping();

  try {
    // Optional hint: if this chat recently had a PDF uploaded, nudge router toward pdfqa
    let hasPdfIndex = false;
    if (activeId) {
      const chatData = await api(`/api/chats/${activeId}`);
      const msgs = (chatData && chatData.messages) || [];
      hasPdfIndex = msgs.some(m => (m.content || "").includes("(uploaded PDF)"));
    }

    // ✅ Call the correct backend route and use the correct fields
    const data = await api("/api/route/intent", {
      method: "POST",
      body: JSON.stringify({ text, has_pdf_index: hasPdfIndex })
    });

    hideTyping();

    const intent = (data && (data.intent || data.tool)) || "(unknown)";
    const conf = data && typeof data.confidence === "number"
      ? data.confidence.toFixed(2)
      : "n/a";
    const method = (data && data.method) || "rules";

    addMessageEl(
      "assistant",
      `**Router choice:** \`${intent}\`\n\n_confidence: ${conf} • via ${method}_`
    );
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ Router failed.");
  }
});

// ===== NEW: Next-Question Prediction =====
nextBtn?.addEventListener("click", async () => {
  addMessageEl("user", "Suggest next questions");
  showTyping();

  try{
    const data = await api("/api/next-questions", {
      method: "POST",
      body: JSON.stringify({ chat_id: activeId || "default" })
    });
    hideTyping();

    const qs = (data && data.suggestions) || [];
    if (!qs.length) { addMessageEl("assistant", "No suggestions available."); return; }

    // render as chips (click to insert)
    const md = qs.map((q,i)=>`**${i+1}.** ${q}`).join("\n\n");
    addMessageEl("assistant", `Here are some suggestions:\n\n${md}`);

    const bar = document.createElement("div");
    bar.style.margin = "8px 0 0";
    bar.innerHTML = qs.map(q => `<button class="btn" style="margin:2px">${q}</button>`).join("");
    chat.appendChild(bar);
    Array.from(bar.querySelectorAll("button")).forEach(btn=>{
      btn.addEventListener("click", () => { input.value = btn.textContent; autosize(); input.focus(); });
    });
    chat.scrollTop = chat.scrollHeight;
  }catch(e){
    hideTyping();
    addMessageEl("assistant", "⚠️ Next-question suggestion failed.");
  }
});

function renderSourceFeedback(sources, label = "Sources used") {
  // normalize to a unique list of source names
  const uniq = [...new Set((sources || []).map(s => {
    if (!s) return null;
    if (typeof s === "string") return s;
    return s.source || null;
  }).filter(Boolean))];

  if (!uniq.length) return;

  const bar = document.createElement("div");
  bar.className = "feedback-bar";
  bar.style.margin = "8px 0 14px";
  bar.style.display = "flex";
  bar.style.flexWrap = "wrap";
  bar.style.gap = "8px";
  bar.style.alignItems = "center";

  const title = document.createElement("div");
  title.textContent = label + ":";
  title.style.opacity = "0.8";
  bar.appendChild(title);

  uniq.forEach(src => {
    const group = document.createElement("div");
    group.style.display = "inline-flex";
    group.style.alignItems = "center";
    group.style.gap = "4px";
    group.style.border = "1px solid var(--border,#333)";
    group.style.borderRadius = "8px";
    group.style.padding = "2px 6px";

    const name = document.createElement("span");
    name.textContent = (src.length > 28 ? src.slice(0, 25) + "…" : src);

    const up = document.createElement("button");
    up.className = "btn";
    up.textContent = "👍";
    up.style.padding = "2px 6px";

    const down = document.createElement("button");
    down.className = "btn";
    down.textContent = "👎";
    down.style.padding = "2px 6px";

    up.addEventListener("click", async () => {
      try {
        await api("/api/feedback/click", {
          method: "POST",
          body: JSON.stringify({ chat_id: activeId || "default", source: src, positive: true })
        });
        up.textContent = "✅";
        setTimeout(() => (up.textContent = "👍"), 900);
      } catch {}
    });

    down.addEventListener("click", async () => {
      try {
        await api("/api/feedback/click", {
          method: "POST",
          body: JSON.stringify({ chat_id: activeId || "default", source: src, positive: false })
        });
        down.textContent = "✅";
        setTimeout(() => (down.textContent = "👎"), 900);
      } catch {}
    });

    group.appendChild(up);
    group.appendChild(down);
    group.appendChild(name);
    bar.appendChild(group);
  });

  chat.appendChild(bar);
  chat.scrollTop = chat.scrollHeight;
}

// ===== INIT =====
(async function init(){
  await loadChats();
})();
