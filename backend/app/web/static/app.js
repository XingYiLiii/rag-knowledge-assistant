const apiBase = "/api/v1";

function pageMessage(text = "", isSuccess = false) {
  const target = document.querySelector("#page-message");
  if (!target) return;
  target.textContent = text;
  target.classList.toggle("success", isSuccess);
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (response.status === 204) return null;
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || "Request failed.");
  return payload;
}

function knowledgeBaseCard(knowledgeBase) {
  const link = document.createElement("a");
  link.className = "card knowledge-base-card";
  link.href = `/knowledge-bases/${knowledgeBase.id}`;
  const title = document.createElement("h3");
  title.textContent = knowledgeBase.name;
  const description = document.createElement("p");
  description.className = "muted";
  description.textContent = knowledgeBase.description || "No description";
  const count = document.createElement("p");
  count.textContent = `${knowledgeBase.document_count} document(s)`;
  link.append(title, description, count);
  return link;
}

async function loadKnowledgeBases() {
  const list = document.querySelector("#knowledge-base-list");
  if (!list) return;
  const knowledgeBases = await request("/knowledge-bases");
  list.replaceChildren();
  if (!knowledgeBases.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No knowledge bases yet.";
    list.append(empty);
    return;
  }
  knowledgeBases.forEach((item) => list.append(knowledgeBaseCard(item)));
}

function initKnowledgeBasesPage() {
  const form = document.querySelector("#create-knowledge-base-form");
  document.querySelector("#refresh-knowledge-bases")?.addEventListener("click", () => loadKnowledgeBases().catch(showError));
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
    } catch (error) { showError(error); }
  });
  loadKnowledgeBases().catch(showError);
}

function documentRow(documentItem) {
  const row = document.createElement("tr");
  const cells = [documentItem.original_name, documentItem.file_type, documentItem.status, String(documentItem.chunk_count), documentItem.error_message || "—"];
  cells.forEach((value, index) => {
    const cell = document.createElement("td");
    if (index === 2) {
      const badge = document.createElement("span");
      badge.className = `status status-${value}`;
      badge.textContent = value;
      cell.append(badge);
    } else cell.textContent = value;
    row.append(cell);
  });
  const actionCell = document.createElement("td");
  actionCell.className = "actions";
  const reindex = document.createElement("button");
  reindex.className = "button-secondary button-small";
  reindex.textContent = "Reindex";
  reindex.addEventListener("click", () => reindexDocument(documentItem.id));
  const remove = document.createElement("button");
  remove.className = "button-danger button-small";
  remove.textContent = "Delete";
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
  } catch (error) { showError(error); }
}

async function reindexDocument(documentId) {
  try {
    await request(`/documents/${documentId}/reindex`, { method: "POST" });
    pageMessage("Document reindexing started.", true);
    await loadKnowledgeBaseDetail();
  } catch (error) { showError(error); }
}

function initKnowledgeBaseDetailPage() {
  document.querySelector("#upload-document-form")?.addEventListener("submit", (event) => uploadDocument(event).catch(showError));
  document.querySelector("#refresh-documents")?.addEventListener("click", () => loadKnowledgeBaseDetail().catch(showError));
  loadKnowledgeBaseDetail().catch(showError);
}

function showError(error) { pageMessage(error instanceof Error ? error.message : "Request failed."); }

if (document.body.dataset.page === "knowledge-bases") initKnowledgeBasesPage();
if (document.body.dataset.page === "knowledge-base-detail") initKnowledgeBaseDetailPage();
