const apiBase = "/api/v1";
const chatState = { conversationId: null };

function pageMessage(text = "", isSuccess = false) {
  const target = document.querySelector("#page-message");
  if (!target) return;
  target.textContent = text;
  target.classList.toggle("success", isSuccess);
}

function showError(error) {
  pageMessage(error instanceof Error ? error.message : "Request failed.");
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (response.status === 204) return null;
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || "Request failed.");
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
    textElement("p", knowledgeBase.description || "No description", "muted"),
    textElement("p", `${knowledgeBase.document_count} document(s)`),
  );
  return link;
}

async function loadKnowledgeBases() {
  const list = document.querySelector("#knowledge-base-list");
  if (!list) return;
  const knowledgeBases = await request("/knowledge-bases");
  list.replaceChildren();
  if (!knowledgeBases.length) {
    list.append(textElement("p", "No knowledge bases yet.", "muted"));
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
      const badge = textElement("span", value, `status status-${value}`);
      cell.append(badge);
    } else {
      cell.textContent = value;
    }
    row.append(cell);
  });
  const actionCell = document.createElement("td");
  actionCell.className = "actions";
  const reindex = textElement("button", "Reindex", "button-secondary button-small");
  reindex.addEventListener("click", () => reindexDocument(documentItem.id));
  const remove = textElement("button", "Delete", "button-danger button-small");
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
  document.querySelector("#knowledge-base-description").textContent = knowledgeBase.description || "No description";
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
  pageMessage("Upload accepted. Document is processing in the background.", true);
  await loadKnowledgeBaseDetail();
}

async function deleteDocument(documentId) {
  if (!window.confirm("Delete this document and its indexed vectors?")) return;
  try {
    await request(`/documents/${documentId}`, { method: "DELETE" });
    pageMessage("Document deleted.", true);
    await loadKnowledgeBaseDetail();
  } catch (error) {
    showError(error);
  }
}

async function reindexDocument(documentId) {
  try {
    await request(`/documents/${documentId}/reindex`, { method: "POST" });
    pageMessage("Document reindexing started.", true);
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
    const option = new Option("No knowledge bases available", "");
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
  const locator = citation.page_number ? `page ${citation.page_number}` : citation.section_title || "chunk";
  summary.textContent = `[${citation.citation_id}] ${citation.document_name} · ${locator}`;
  const metadata = textElement("p", `Score: ${Number(citation.score).toFixed(3)}`, "muted");
  const matched = document.createElement("pre");
  matched.className = "matched-text";
  matched.textContent = citation.matched_text;
  details.append(summary, metadata, matched);
  return details;
}

function renderChatMessage(message) {
  const article = document.createElement("article");
  article.className = `chat-message chat-message-${message.role}`;
  article.append(textElement("p", message.role === "assistant" ? "Assistant" : "You", "message-role"));
  article.append(textElement("p", message.content, "message-content"));
  if (message.citations?.length) {
    const citations = document.createElement("div");
    citations.className = "citation-list";
    citations.append(textElement("h3", "Sources", "citation-heading"));
    message.citations.forEach((citation) => citations.append(renderCitation(citation)));
    article.append(citations);
  }
  return article;
}

function renderMessages(target, messages) {
  target.replaceChildren();
  if (!messages.length) {
    target.append(textElement("p", "No messages yet.", "muted"));
    return;
  }
  messages.forEach((message) => target.append(renderChatMessage(message)));
}

async function loadConversationOptions(knowledgeBaseId, select) {
  const conversations = await request(`/knowledge-bases/${knowledgeBaseId}/conversations`);
  select.replaceChildren(new Option("New conversation", ""));
  conversations.forEach((conversation) => {
    const createdAt = new Date(conversation.created_at).toLocaleString();
    select.append(new Option(`Conversation · ${createdAt}`, conversation.id));
  });
  return conversations;
}

async function createConversation(knowledgeBaseId, select) {
  const conversation = await request("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ knowledge_base_id: knowledgeBaseId }),
  });
  const option = new Option("Current conversation", conversation.id);
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
    pageMessage("Create a knowledge base before starting a chat.");
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
    pageMessage("A new conversation will be created when you send a question.", true);
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
      pageMessage(`Answered with ${response.used_chunks} source chunk(s).`, true);
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
    pageMessage("No knowledge bases available.");
    return;
  }

  async function loadHistory() {
    const conversations = await request(`/knowledge-bases/${knowledgeBaseSelect.value}/conversations`);
    conversationList.replaceChildren();
    renderMessages(messageList, []);
    if (!conversations.length) {
      conversationList.append(textElement("p", "No saved conversations.", "muted"));
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