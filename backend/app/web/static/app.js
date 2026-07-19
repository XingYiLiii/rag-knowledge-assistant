const apiBase = "/api/v1";
const chatState = { conversationId: null };
const documentStatusLabels = {
  pending: "等待处理",
  processing: "处理中",
  ready: "已完成",
  failed: "失败",
};

function pageMessage(text = "", isSuccess = false) {
  const target = document.querySelector("#page-message");
  if (!target) return;
  target.textContent = text;
  target.classList.toggle("success", isSuccess);
}

function showError(error) {
  pageMessage(error instanceof Error ? error.message : "请求失败。");
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (response.status === 204) return null;
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || "请求失败。");
  return payload;
}

function textElement(tagName, text, className = "") {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function knowledgeBaseCard(knowledgeBase) {
  const link = document.createElement("a");
  link.className = "card knowledge-base-card";
  link.href = `/knowledge-bases/${knowledgeBase.id}`;
  link.append(
    textElement("h3", knowledgeBase.name),
    textElement("p", knowledgeBase.description || "暂无描述", "muted"),
    textElement("p", `${knowledgeBase.document_count} 个文档`),
  );
  return link;
}

async function loadKnowledgeBases() {
  const list = document.querySelector("#knowledge-base-list");
  if (!list) return;
  const knowledgeBases = await request("/knowledge-bases");
  list.replaceChildren();
  if (!knowledgeBases.length) {
    list.append(textElement("p", "暂无知识库。", "muted"));
    return;
  }
  knowledgeBases.forEach((item) => list.append(knowledgeBaseCard(item)));
}

function initKnowledgeBasesPage() {
  const form = document.querySelector("#create-knowledge-base-form");
  document.querySelector("#refresh-knowledge-bases")?.addEventListener("click", () => {
    loadKnowledgeBases().catch(showError);
  });
  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = document.querySelector("#knowledge-base-name").value;
    const description = document.querySelector("#knowledge-base-description").value;
    try {
      const knowledgeBase = await request("/knowledge-bases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description: description || null }),
      });
      window.location.assign(`/knowledge-bases/${knowledgeBase.id}`);
    } catch (error) {
      showError(error);
    }
  });
  loadKnowledgeBases().catch(showError);
}

function documentRow(documentItem) {
  const row = document.createElement("tr");
  const cells = [
    documentItem.original_name,
    documentItem.file_type,
    documentItem.status,
    String(documentItem.chunk_count),
    documentItem.error_message || "—",
  ];
  cells.forEach((value, index) => {
    const cell = document.createElement("td");
    if (index === 2) {
      const label = documentStatusLabels[value] ?? value;
      const badge = textElement("span", label, `status status-${value}`);
      cell.append(badge);
    } else {
      cell.textContent = value;
    }
    row.append(cell);
  });
  const actionCell = document.createElement("td");
  actionCell.className = "actions";
  const reindex = textElement("button", "重新索引", "button-secondary button-small");
  reindex.addEventListener("click", () => reindexDocument(documentItem.id));
  const remove = textElement("button", "删除", "button-danger button-small");
  remove.addEventListener("click", () => deleteDocument(documentItem.id));
  actionCell.append(reindex, remove);
  row.append(actionCell);
  return row;
}
function currentKnowledgeBaseId() {
  return document.querySelector("[data-knowledge-base-id]")?.dataset.knowledgeBaseId;
}

async function loadKnowledgeBaseDetail() {
  const knowledgeBaseId = currentKnowledgeBaseId();
  if (!knowledgeBaseId) return;
  const [knowledgeBase, documentPage] = await Promise.all([
    request(`/knowledge-bases/${knowledgeBaseId}`),
    request(`/knowledge-bases/${knowledgeBaseId}/documents?page=1&page_size=100`),
  ]);
  document.querySelector("#knowledge-base-name").textContent = knowledgeBase.name;
  document.querySelector("#knowledge-base-description").textContent = knowledgeBase.description || "暂无描述";
  const list = document.querySelector("#document-list");
  list.replaceChildren(...documentPage.items.map(documentRow));
}

async function uploadDocument(event) {
  event.preventDefault();
  const knowledgeBaseId = currentKnowledgeBaseId();
  const input = document.querySelector("#document-file");
  if (!input.files?.[0]) return;
  const formData = new FormData();
  formData.append("file", input.files[0]);
  await request(`/knowledge-bases/${knowledgeBaseId}/documents`, { method: "POST", body: formData });
  input.value = "";
  pageMessage("上传成功，文档正在后台处理中。", true);
  await loadKnowledgeBaseDetail();
}

async function deleteDocument(documentId) {
  if (!window.confirm("确定删除该文档及其已索引的向量吗？")) return;
  try {
    await request(`/documents/${documentId}`, { method: "DELETE" });
    pageMessage("文档已删除。", true);
    await loadKnowledgeBaseDetail();
  } catch (error) {
    showError(error);
  }
}

async function reindexDocument(documentId) {
  try {
    await request(`/documents/${documentId}/reindex`, { method: "POST" });
    pageMessage("文档已开始重新索引。", true);
    await loadKnowledgeBaseDetail();
  } catch (error) {
    showError(error);
  }
}

function initKnowledgeBaseDetailPage() {
  document.querySelector("#upload-document-form")?.addEventListener("submit", (event) => {
    uploadDocument(event).catch(showError);
  });
  document.querySelector("#refresh-documents")?.addEventListener("click", () => {
    loadKnowledgeBaseDetail().catch(showError);
  });
  loadKnowledgeBaseDetail().catch(showError);
}

function populateKnowledgeBaseSelect(select, knowledgeBases) {
  select.replaceChildren();
  if (!knowledgeBases.length) {
    const option = new Option("暂无可用知识库", "");
    option.disabled = true;
    option.selected = true;
    select.append(option);
    return;
  }
  knowledgeBases.forEach((knowledgeBase) => {
    select.append(new Option(knowledgeBase.name, knowledgeBase.id));
  });
}

function renderCitation(citation) {
  const details = document.createElement("details");
  details.className = "citation-card";
  const summary = document.createElement("summary");
  const locator = citation.page_number ? `第 ${citation.page_number} 页` : citation.section_title || "切片";
  summary.textContent = `[${citation.citation_id}] ${citation.document_name} ｜ ${locator}`;
  const metadata = textElement("p", `匹配分数：${Number(citation.score).toFixed(3)}`, "muted");
  const matched = document.createElement("pre");
  matched.className = "matched-text";
  matched.textContent = citation.matched_text;
  details.append(summary, metadata, matched);
  return details;
}

function renderChatMessage(message) {
  const article = document.createElement("article");
  article.className = `chat-message chat-message-${message.role}`;
  article.append(textElement("p", message.role === "assistant" ? "助手" : "你", "message-role"));
  article.append(textElement("p", message.content, "message-content"));
  if (message.citations?.length) {
    const citations = document.createElement("div");
    citations.className = "citation-list";
    citations.append(textElement("h3", "引用来源", "citation-heading"));
    message.citations.forEach((citation) => citations.append(renderCitation(citation)));
    article.append(citations);
  }
  return article;
}

function renderMessages(target, messages) {
  target.replaceChildren();
  if (!messages.length) {
    target.append(textElement("p", "暂无消息。", "muted"));
    return;
  }
  messages.forEach((message) => target.append(renderChatMessage(message)));
}

async function loadConversationOptions(knowledgeBaseId, select) {
  const conversations = await request(`/knowledge-bases/${knowledgeBaseId}/conversations`);
  select.replaceChildren(new Option("新建对话", ""));
  conversations.forEach((conversation) => {
    const createdAt = new Date(conversation.created_at).toLocaleString();
    select.append(new Option(`对话 ｜ ${createdAt}`, conversation.id));
  });
  return conversations;
}

async function createConversation(knowledgeBaseId, select) {
  const conversation = await request("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ knowledge_base_id: knowledgeBaseId }),
  });
  const option = new Option("当前对话", conversation.id);
  select.append(option);
  select.value = conversation.id;
  chatState.conversationId = conversation.id;
  return conversation.id;
}

async function loadConversationMessages(conversationId, target) {
  const payload = await request(`/conversations/${conversationId}/messages`);
  renderMessages(target, payload.messages);
}

async function initChatPage() {
  const knowledgeBaseSelect = document.querySelector("#chat-knowledge-base");
  const conversationSelect = document.querySelector("#chat-conversation");
  const messageList = document.querySelector("#chat-messages");
  const form = document.querySelector("#chat-form");
  const question = document.querySelector("#chat-question");
  const knowledgeBases = await request("/knowledge-bases");
  populateKnowledgeBaseSelect(knowledgeBaseSelect, knowledgeBases);
  if (!knowledgeBases.length) {
    pageMessage("请先创建知识库，再开始智能问答。");
    return;
  }

  async function refreshConversations() {
    chatState.conversationId = null;
    await loadConversationOptions(knowledgeBaseSelect.value, conversationSelect);
    renderMessages(messageList, []);
  }

  knowledgeBaseSelect.addEventListener("change", () => refreshConversations().catch(showError));
  conversationSelect.addEventListener("change", () => {
    chatState.conversationId = conversationSelect.value || null;
    if (chatState.conversationId) {
      loadConversationMessages(chatState.conversationId, messageList).catch(showError);
    } else {
      renderMessages(messageList, []);
    }
  });
  document.querySelector("#new-conversation")?.addEventListener("click", () => {
    chatState.conversationId = null;
    conversationSelect.value = "";
    renderMessages(messageList, []);
    pageMessage("发送问题后将自动创建新对话。", true);
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = question.value.trim();
    if (!value) return;
    try {
      if (!chatState.conversationId) {
        await createConversation(knowledgeBaseSelect.value, conversationSelect);
      }
      const response = await request("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          knowledge_base_id: knowledgeBaseSelect.value,
          conversation_id: chatState.conversationId,
          question: value,
        }),
      });
      messageList.append(renderChatMessage({ role: "user", content: value }));
      messageList.append(renderChatMessage({ role: "assistant", content: response.answer, citations: response.citations }));
      question.value = "";
      pageMessage(`已基于 ${response.used_chunks} 个来源切片生成回答。`, true);
    } catch (error) {
      showError(error);
    }
  });
  await refreshConversations();
}

function conversationButton(conversation, onSelect) {
  const button = textElement("button", "", "conversation-button button-secondary");
  button.type = "button";
  button.textContent = new Date(conversation.updated_at).toLocaleString();
  button.addEventListener("click", () => onSelect(conversation.id));
  return button;
}

async function initHistoryPage() {
  const knowledgeBaseSelect = document.querySelector("#history-knowledge-base");
  const conversationList = document.querySelector("#conversation-list");
  const messageList = document.querySelector("#history-messages");
  const knowledgeBases = await request("/knowledge-bases");
  populateKnowledgeBaseSelect(knowledgeBaseSelect, knowledgeBases);
  if (!knowledgeBases.length) {
    pageMessage("暂无可用知识库。");
    return;
  }

  async function loadHistory() {
    const conversations = await request(`/knowledge-bases/${knowledgeBaseSelect.value}/conversations`);
    conversationList.replaceChildren();
    renderMessages(messageList, []);
    if (!conversations.length) {
      conversationList.append(textElement("p", "暂无已保存的对话。", "muted"));
      return;
    }
    conversations.forEach((conversation) => {
      conversationList.append(conversationButton(conversation, (conversationId) => {
        loadConversationMessages(conversationId, messageList).catch(showError);
      }));
    });
  }

  knowledgeBaseSelect.addEventListener("change", () => loadHistory().catch(showError));
  await loadHistory();
}

if (document.body.dataset.page === "knowledge-bases") initKnowledgeBasesPage();
if (document.body.dataset.page === "knowledge-base-detail") initKnowledgeBaseDetailPage();
if (document.body.dataset.page === "chat") initChatPage().catch(showError);
if (document.body.dataset.page === "history") initHistoryPage().catch(showError);