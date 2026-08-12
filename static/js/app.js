// ============================================================
// app.js – полная клиентская логика Smart Brain v10
// ============================================================

// ---------- Состояние ----------
const STORAGE_KEY = 'brain_chats';
const SETTINGS_KEY = 'brain_settings';
const MOBILE_QUERY = '(max-width: 900px)';

let chats = [];
let currentChatId = null;
let lastQuestion = '';
let lastAnswer = '';
let lastFacts = [];
let chatListCollapsed = false;
let settingsOpen = false;
let interactiveMode = false;
let currentAgentQuestion = null;
let pollIntervalId = null;
let agentRunning = true;
let searchEnabled = false;

// ---------- DOM refs ----------
const $ = id => document.getElementById(id);
const messagesEl = $('messages');
const questionInput = $('questionInput');
const sendBtn = $('sendBtn');
const chatItemsEl = $('chatItems');
const tempSlider = $('tempSlider');
const tempDisplay = $('tempDisplay');
const allowClarifying = $('allowClarifying');
const clarifyingDelay = $('clarifyingDelay');
const delayDisplay = $('delayDisplay');
const interactiveModeChk = $('interactiveMode');
const userTimeout = $('userTimeout');
const applyAgentConfigBtn = $('applyAgentConfig');
const interactiveLog = $('interactiveLog');
const trainTopicBtn = $('trainTopicBtn');
const trainLog = $('trainLog');
const trainPairBtn = $('trainPairBtn');
const pairLog = $('pairLog');
const sleepBtn = $('sleepBtn');
const sleepLog = $('sleepLog');
const agentToggleBtn = $('agentToggleBtn');
const agentLog = $('agentLog');
const updateAgentConfigBtn = $('updateAgentConfig');
const learnPosBtn = $('learnPosBtn');
const learnNegBtn = $('learnNegBtn');
const statsContainer = $('statsContainer');
const factsList = $('factsList');
const settingsPanel = $('settingsPanel');
const chatList = $('chatList');
const overlayEl = $('overlay');
const searchToggleBtn = $('searchToggleBtn');

// ---------- Вспомогательные функции ----------
function stripHtml(html) {
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.textContent || '';
}

function isMobile() {
    return window.matchMedia(MOBILE_QUERY).matches;
}

// ---------- Инициализация ----------
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadChats();
    loadStats();
    bindEvents();
    const chat = getCurrentChat();
    if (chat && chat.messages.length === 0) {
        addMessage('Привет! Я Smart Brain. Задай мне вопрос.', 'bot', null, false, false);
    }
    setInterval(loadStats, 15000);
    questionInput.focus();
    autoResizeTextarea();
});

// ---------- События ----------
function bindEvents() {
    sendBtn.addEventListener('click', sendMessage);
    questionInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            e.preventDefault();
            sendMessage();
        }
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            sendMessage();
        }
    });
    questionInput.addEventListener('input', () => {
        autoResizeTextarea();
        sendBtn.disabled = questionInput.value.trim() === '';
    });

    tempSlider.addEventListener('input', () => {
        tempDisplay.textContent = tempSlider.value;
        saveSettings();
    });
    tempSlider.addEventListener('change', saveSettings);

    clarifyingDelay.addEventListener('input', () => {
        delayDisplay.textContent = clarifyingDelay.value;
        saveSettings();
    });
    clarifyingDelay.addEventListener('change', saveSettings);

    [allowClarifying, interactiveModeChk, userTimeout].forEach(el => {
        el.addEventListener('change', saveSettings);
    });
    interactiveModeChk.addEventListener('change', toggleInteractiveMode);

    document.getElementById('toggleChatList').addEventListener('click', toggleChatList);
    document.getElementById('toggleSettings').addEventListener('click', toggleSettings);
    document.getElementById('newChatBtn').addEventListener('click', createNewChat);
    learnPosBtn.addEventListener('click', () => learnLastPair('positive'));
    learnNegBtn.addEventListener('click', () => learnLastPair('negative'));
    searchToggleBtn.addEventListener('click', () => {
        searchEnabled = !searchEnabled;
        searchToggleBtn.classList.toggle('active', searchEnabled);
        searchToggleBtn.textContent = searchEnabled ? '🌐' : '🌐';
    });
    applyAgentConfigBtn.addEventListener('click', updateAgentInteractiveConfig);
    trainTopicBtn.addEventListener('click', trainTopic);
    trainPairBtn.addEventListener('click', trainPair);
    sleepBtn.addEventListener('click', sleepBrain);
    agentToggleBtn.addEventListener('click', toggleAgent);
    updateAgentConfigBtn.addEventListener('click', updateAgentConfig);

    if (overlayEl) {
        overlayEl.addEventListener('click', closeMobilePanels);
    }
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeMobilePanels();
    });

    document.querySelectorAll('#settingsPanel input, #settingsPanel select').forEach(el => {
        if (!['tempSlider', 'clarifyingDelay', 'interactiveMode', 'allowClarifying', 'userTimeout'].includes(el.id)) {
            el.addEventListener('change', saveSettings);
            if (el.type === 'range') el.addEventListener('input', saveSettings);
        }
    });
    document.getElementById('agentTopics').addEventListener('blur', saveSettings);
    document.getElementById('agentInterval').addEventListener('blur', saveSettings);
    document.getElementById('agentQCount').addEventListener('blur', saveSettings);
}

// ---------- Авто-растягивание textarea ----------
function autoResizeTextarea() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 150) + 'px';
    if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
}

// ---------- Управление настройками ----------
// ---------- Управление настройками ----------
function loadSettings() {
    const stored = localStorage.getItem(SETTINGS_KEY);
    let hadStoredChatListState = false;
    if (stored) {
        try {
            const s = JSON.parse(stored);
            if (s.temperature !== undefined) {
                tempSlider.value = s.temperature;
                tempDisplay.textContent = s.temperature;
            }
            if (s.allowClarifying !== undefined) allowClarifying.checked = s.allowClarifying;
            if (s.clarifyingDelay !== undefined) {
                clarifyingDelay.value = s.clarifyingDelay;
                delayDisplay.textContent = s.clarifyingDelay;
            }
            if (s.interactiveMode !== undefined) {
                interactiveModeChk.checked = s.interactiveMode;
                interactiveMode = s.interactiveMode;
                if (s.interactiveMode) startPolling();
            }
            if (s.userTimeout !== undefined) userTimeout.value = s.userTimeout;
            if (s.agentTopics !== undefined) document.getElementById('agentTopics').value = s.agentTopics;
            if (s.agentInterval !== undefined) document.getElementById('agentInterval').value = s.agentInterval;
            if (s.agentQCount !== undefined) document.getElementById('agentQCount').value = s.agentQCount;
            // НОВЫЕ ПОЛЯ
            if (s.agentTeacherThreshold !== undefined) document.getElementById('agentTeacherThreshold').value = s.agentTeacherThreshold;
            if (s.agentUseDynamicThreshold !== undefined) document.getElementById('agentUseDynamicThreshold').checked = s.agentUseDynamicThreshold;
            // Для bulk train тоже можно сохранять, но это не критично
            if (s.bulkSleepEvery !== undefined) document.getElementById('bulkSleepEvery').value = s.bulkSleepEvery;
            if (s.bulkReward !== undefined) document.getElementById('bulkReward').value = s.bulkReward;
            if (s.bulkValidate !== undefined) document.getElementById('bulkValidate').checked = s.bulkValidate;
            if (s.bulkThreshold !== undefined) document.getElementById('bulkThreshold').value = s.bulkThreshold;
            if (s.bulkRetries !== undefined) document.getElementById('bulkRetries').value = s.bulkRetries;

            if (s.chatListCollapsed !== undefined) {
                hadStoredChatListState = true;
                chatListCollapsed = s.chatListCollapsed;
            }
            if (s.settingsOpen !== undefined) {
                settingsOpen = s.settingsOpen;
            }
        } catch(e) { console.warn('Settings load error', e); }
    }

    if (!hadStoredChatListState && isMobile()) {
        chatListCollapsed = true;
    }

    chatList.classList.toggle('collapsed', chatListCollapsed);
    settingsPanel.classList.toggle('open', settingsOpen);
    updateOverlay();
}

function saveSettings() {
    const settings = {
        temperature: parseFloat(tempSlider.value),
        allowClarifying: allowClarifying.checked,
        clarifyingDelay: parseFloat(clarifyingDelay.value),
        interactiveMode: interactiveModeChk.checked,
        userTimeout: parseInt(userTimeout.value) || 30,
        agentTopics: document.getElementById('agentTopics').value,
        agentInterval: parseInt(document.getElementById('agentInterval').value) || 120,
        agentQCount: parseInt(document.getElementById('agentQCount').value) || 2,
        // НОВЫЕ ПОЛЯ
        agentTeacherThreshold: parseFloat(document.getElementById('agentTeacherThreshold').value) || 0.7,
        agentUseDynamicThreshold: document.getElementById('agentUseDynamicThreshold').checked,
        // Для bulk train (опционально)
        bulkSleepEvery: parseInt(document.getElementById('bulkSleepEvery').value) || 200,
        bulkReward: parseFloat(document.getElementById('bulkReward').value) || 1.0,
        bulkValidate: document.getElementById('bulkValidate').checked,
        bulkThreshold: parseFloat(document.getElementById('bulkThreshold').value) || 0.6,
        bulkRetries: parseInt(document.getElementById('bulkRetries').value) || 1,
        chatListCollapsed: chatListCollapsed,
        settingsOpen: settingsOpen
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

// ---------- Чаты ----------
function loadChats() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        try { chats = JSON.parse(stored); } catch(e) { chats = []; }
    }
    if (!chats.length) {
        const id = Date.now().toString();
        chats = [{ id, name: 'Новый чат', messages: [] }];
        currentChatId = id;
        saveChats();
    } else {
        currentChatId = chats[0].id;
    }
    renderChatList();
    loadCurrentChat();
}

function saveChats() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

function getCurrentChat() {
    return chats.find(c => c.id === currentChatId);
}

function renderChatList() {
    chatItemsEl.innerHTML = '';
    chats.forEach(chat => {
        const div = document.createElement('div');
        div.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
        const nameSpan = document.createElement('span');
        nameSpan.className = 'chat-name';
        nameSpan.textContent = chat.name;
        nameSpan.ondblclick = (e) => {
            e.stopPropagation();
            const newName = prompt('Введите новое имя чата:', chat.name);
            if (newName && newName.trim()) {
                chat.name = newName.trim();
                saveChats();
                renderChatList();
            }
        };
        const delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '✕';
        delBtn.onclick = (e) => {
            e.stopPropagation();
            deleteChat(chat.id);
        };
        div.appendChild(nameSpan);
        div.appendChild(delBtn);
        div.onclick = () => switchChat(chat.id);
        chatItemsEl.appendChild(div);
    });
}

function switchChat(chatId) {
    if (chatId === currentChatId) {
        if (isMobile()) closeMobilePanels();
        return;
    }
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;
    currentChatId = chatId;
    renderChatList();
    loadCurrentChat();
    if (isMobile()) closeMobilePanels();
}

function loadCurrentChat() {
    const chat = getCurrentChat();
    if (!chat) return;
    messagesEl.innerHTML = '';
    if (chat.messages.length === 0) {
        addMessage('Привет! Я Smart Brain. Задай мне вопрос.', 'bot', null, false, false);
    } else {
        chat.messages.forEach(msg => {
            const div = document.createElement('div');
            div.className = 'message ' + msg.role;
            if (msg.isAgentOrClarifying) div.classList.add('agent-question');
            div.innerHTML = msg.text;
            messagesEl.appendChild(div);
        });
    }
    highlightCodeBlocks();

    // Поиск последней пары
    lastQuestion = '';
    lastAnswer = '';
    lastFacts = [];
    const msgs = chat.messages;
    for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === 'user') {
            lastQuestion = stripHtml(msgs[i].text);
            for (let j = i + 1; j < msgs.length; j++) {
                if (msgs[j].role === 'bot' && !msgs[j].isAgentOrClarifying) {
                    lastAnswer = stripHtml(msgs[j].text);
                    lastFacts = msgs[j].facts || [];
                    break;
                }
            }
            break;
        }
    }
    const hasPair = !!(lastQuestion && lastAnswer);
    learnPosBtn.style.display = hasPair ? 'inline-flex' : 'none';
    learnNegBtn.style.display = hasPair ? 'inline-flex' : 'none';
    searchToggleBtn.style.display = hasPair ? 'inline-flex' : 'none';
    renderFacts(lastFacts);

    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function createNewChat() {
    const id = Date.now().toString();
    const name = 'Новый чат';
    chats.push({ id, name, messages: [] });
    saveChats();
    switchChat(id);
}

let pendingDeleteChatId = null;

function showConfirm(message, onConfirm) {
    const modal = document.getElementById('confirmModal');
    const msg = document.getElementById('confirmMessage');
    const okBtn = document.getElementById('confirmOkBtn');
    const cancelBtn = document.getElementById('confirmCancelBtn');
    msg.textContent = message;
    modal.style.display = 'flex';
    const newOk = okBtn.cloneNode(true);
    const newCancel = cancelBtn.cloneNode(true);
    okBtn.parentNode.replaceChild(newOk, okBtn);
    cancelBtn.parentNode.replaceChild(newCancel, cancelBtn);
    newOk.addEventListener('click', () => {
        modal.style.display = 'none';
        onConfirm(true);
    });
    newCancel.addEventListener('click', () => {
        modal.style.display = 'none';
        onConfirm(false);
    });
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
            onConfirm(false);
        }
    });
}

function deleteChat(chatId) {
    if (chats.length <= 1) {
        showToast('⚠️ Нельзя удалить последний чат.', 'error');
        return;
    }
    pendingDeleteChatId = chatId;
    showConfirm('Удалить чат?', (confirmed) => {
        if (!confirmed) return;
        const idx = chats.findIndex(c => c.id === pendingDeleteChatId);
        if (idx === -1) return;
        chats.splice(idx, 1);
        if (currentChatId === pendingDeleteChatId) currentChatId = chats[0].id;
        saveChats();
        renderChatList();
        loadCurrentChat();
        showToast('🗑️ Чат удалён', 'success');
        pendingDeleteChatId = null;
    });
}

function autoRenameChat(chatId, firstMessage) {
    const chat = chats.find(c => c.id === chatId);
    if (!chat || chat.name !== 'Новый чат') return;
    let newName = firstMessage.trim();
    if (!newName) return;
    newName = newName.replace(/\d{1,2}:\d{2}(:\d{2})?/g, '').trim();
    const words = newName.split(/\s+/);
    let name = words.slice(0, 6).join(' ');
    if (name.length > 40) name = name.substring(0, 40) + '…';
    chat.name = name || 'Новый чат';
    saveChats();
    renderChatList();
}

// ---------- Сообщения ----------
function addMessage(text, sender, facts, isAgentOrClarifying, animate = false) {
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    if (isAgentOrClarifying) div.classList.add('agent-question');

    const factsHtml = (sender === 'bot' && facts && facts.length) ?
        '<div class="facts-indicator">📚 использовано фактов: ' + facts.length + '</div>' : '';
    const timeHtml = '<div class="time">' + new Date().toLocaleTimeString() + '</div>';

    if (sender === 'user' || !animate) {
        const formatted = formatMessage(text);
        div.innerHTML = formatted + factsHtml + timeHtml;
        messagesEl.appendChild(div);
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
        highlightCodeBlocks();
        saveMessageToChat(div.innerHTML, sender, facts, isAgentOrClarifying);
    } else {
        div.innerHTML = '';
        messagesEl.appendChild(div);
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });

        const words = text.split(/(\s+)/);
        let fullText = '', idx = 0;
        function typeNext() {
            if (!div.parentNode) return;
            if (idx < words.length) {
                fullText += words[idx++];
                div.textContent = fullText;
                requestAnimationFrame(() => {
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                });
                setTimeout(typeNext, 10 + Math.random() * 20);
            } else {
                const formatted = formatMessage(fullText);
                div.innerHTML = formatted + factsHtml + timeHtml;
                requestAnimationFrame(() => {
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                });
                highlightCodeBlocks();
                saveMessageToChat(div.innerHTML, sender, facts, isAgentOrClarifying);
            }
        }
        typeNext();
    }
}

// ---------- НОВАЯ ФУНКЦИЯ: сообщение с мыслями и кратким ответом ----------
function addMessageWithThoughts(shortAnswer, thoughts, facts, sender = 'bot', isAgentOrClarifying = false, animate = false) {
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    if (isAgentOrClarifying) div.classList.add('agent-question');

    // Блок мыслей (если есть и отличается от краткого ответа)
    if (thoughts && thoughts.trim() && thoughts.trim() !== shortAnswer.trim()) {
        const details = document.createElement('details');
        details.className = 'thoughts-details';
        const summary = document.createElement('summary');
        summary.textContent = '🧠 Мысли (нажмите, чтобы развернуть)';
        details.appendChild(summary);

        const thoughtsDiv = document.createElement('div');
        thoughtsDiv.className = 'thinking-text';
        thoughtsDiv.textContent = thoughts;
        details.appendChild(thoughtsDiv);

        div.appendChild(details);
    }

    // Контейнер для краткого ответа
    const answerContainer = document.createElement('div');
    answerContainer.className = 'answer-text';
    div.appendChild(answerContainer);

    const factsHtml = (facts && facts.length) ? `<div class="facts-indicator">📚 использовано фактов: ${facts.length}</div>` : '';
    const timeHtml = `<div class="time">${new Date().toLocaleTimeString()}</div>`;

    if (!animate) {
        answerContainer.innerHTML = formatMessage(shortAnswer);
        div.innerHTML += factsHtml + timeHtml;
        messagesEl.appendChild(div);
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
        highlightCodeBlocks();
        saveMessageToChat(div.innerHTML, sender, facts, isAgentOrClarifying);
    } else {
        messagesEl.appendChild(div);
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });

        const words = shortAnswer.split(/(\s+)/);
        let fullText = '', idx = 0;
        function typeNext() {
            if (!div.parentNode) return;
            if (idx < words.length) {
                fullText += words[idx++];
                answerContainer.textContent = fullText;
                requestAnimationFrame(() => {
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                });
                setTimeout(typeNext, 10 + Math.random() * 20);
            } else {
                answerContainer.innerHTML = formatMessage(fullText);
                // Добавляем факты и время
                const extra = document.createElement('div');
                extra.innerHTML = factsHtml + timeHtml;
                div.appendChild(extra);
                requestAnimationFrame(() => {
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                });
                highlightCodeBlocks();
                saveMessageToChat(div.innerHTML, sender, facts, isAgentOrClarifying);
            }
        }
        typeNext();
    }
}

// ---------- Сохранение сообщения в чат ----------
function saveMessageToChat(html, sender, facts, isAgentOrClarifying) {
    const chat = getCurrentChat();
    if (!chat) return;
    if (sender === 'user' && !chat.messages.some(m => m.role === 'user')) {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        const text = tempDiv.textContent || '';
        autoRenameChat(chat.id, text);
    }
    chat.messages.push({
        role: sender,
        text: html,
        facts: facts || null,
        isAgentOrClarifying: isAgentOrClarifying || false,
        time: new Date().toISOString()
    });
    saveChats();
}

// ---------- Форматирование сообщений (маркдаун, код) ----------
function formatMessage(text) {
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let parts = [];
    let lastIndex = 0;
    let match;
    while ((match = codeBlockRegex.exec(text)) !== null) {
        const lang = match[1] || 'text';
        const code = match[2];
        const before = text.substring(lastIndex, match.index);
        if (before) parts.push({ type: 'text', content: before });
        parts.push({ type: 'code', lang, code });
        lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
        parts.push({ type: 'text', content: text.substring(lastIndex) });
    }

    function processText(content) {
        let escaped = escapeHtml(content);
        escaped = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        escaped = escaped.replace(/\*(.+?)\*/g, '<em>$1</em>');
        const paragraphs = escaped.split(/\n\s*\n/);
        return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    }

    let result = '';
    for (const part of parts) {
        if (part.type === 'code') {
            result += `<pre><code class="language-${part.lang}">${escapeHtml(part.code)}</code><button class="copy-btn" onclick="window.copyCode(this)">Копировать</button></pre>`;
        } else {
            result += processText(part.content);
        }
    }
    return result;
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

window.copyCode = function(btn) {
    const pre = btn.parentElement;
    const code = pre.querySelector('code');
    if (code) {
        navigator.clipboard.writeText(code.textContent).then(() => {
            btn.textContent = '✅';
            setTimeout(() => btn.textContent = 'Копировать', 2000);
        }).catch(() => {});
    }
};

function highlightCodeBlocks() {
    document.querySelectorAll('.message pre code').forEach(block => {
        if (!block.dataset.highlighted) {
            hljs.highlightElement(block);
            block.dataset.highlighted = 'true';
        }
    });
}

// ---------- Отправка сообщения (ОСНОВНАЯ ФУНКЦИЯ) ----------
async function sendMessage() {
    const text = questionInput.value.trim();
    if (!text) return;
    questionInput.value = '';
    autoResizeTextarea();
    sendBtn.disabled = true;
    questionInput.focus();

    // ===== РЕЖИМ АГЕНТА =====
    if (interactiveMode && currentAgentQuestion) {
        const q = currentAgentQuestion;
        currentAgentQuestion = null;
        interactiveLog.textContent = 'Ответ отправлен агенту.';
        try {
            await fetch('/agent/submit_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, answer: text })
            });
            addMessage(text, 'user');
            addMessage('✅ Ответ принят и изучен.', 'bot', null, false, false);
            loadStats();
        } catch(e) {
            addMessage('❌ Ошибка при отправке ответа: ' + e.message, 'bot', null, false, false);
        } finally {
            sendBtn.disabled = false;
        }
        return;
    }

    // ===== ОБЫЧНЫЙ РЕЖИМ =====
    addMessage(text, 'user', null, false, false);

    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing';
    typingDiv.id = typingId;
    typingDiv.innerHTML = '<span class="dots">думает</span>';
    messagesEl.appendChild(typingDiv);
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });

    sendBtn.disabled = true;

    try {
        const temp = parseFloat(tempSlider.value);
        const allowClar = allowClarifying.checked;
        const useSearch = searchEnabled;

        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: text,
                temperature: temp,
                allow_clarifying: allowClar,
                use_search: useSearch
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Ошибка');
        }

        const data = await response.json();

        // ---- Извлекаем краткий ответ и полные мысли ----
        const shortAnswer = data.answer || 'Нет ответа.';
        const thoughts = data.thoughts || shortAnswer;  // если thoughts нет, используем shortAnswer
        const finalFacts = data.facts || [];

        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // ---- Добавляем сообщение с мыслями и кратким ответом ----
        addMessageWithThoughts(shortAnswer, thoughts, finalFacts, 'bot', false, true);

        // ---- Уточняющий вопрос (если есть) ----
        if (data.clarifying_question && allowClar) {
            const delay = parseFloat(clarifyingDelay.value) * 1000;
            setTimeout(() => {
                addMessage('🤔 ' + data.clarifying_question, 'bot', null, true, false);
            }, delay);
        }

        // ---- Обновляем состояние ----
        lastQuestion = text;
        lastAnswer = stripHtml(shortAnswer);
        lastFacts = finalFacts;
        updateButtons();
        renderFacts(lastFacts);
        loadStats();

        searchEnabled = false;
        searchToggleBtn.classList.remove('active');

    } catch (e) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        addMessage('❌ Ошибка: ' + e.message, 'bot', null, false, false);
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

// ---------- Обучение пары ----------
async function learnLastPair(type) {
    if (!lastQuestion || !lastAnswer) {
        showToast('Нет пары для обучения.', 'error');
        return;
    }

    const btn = type === 'positive' ? learnPosBtn : learnNegBtn;
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳...';
    showToast('Обучение...', 'info', 0);

    try {
        const url = type === 'negative' ? '/learn_neg' : '/learn';
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: lastQuestion, answer: lastAnswer })
        });

        if (res.ok) {
            showToast(`✅ ${type === 'positive' ? 'Положительное' : 'Отрицательное'} обучение успешно!`, 'success');
            loadStats();
            learnPosBtn.style.display = 'none';
            learnNegBtn.style.display = 'none';
        } else {
            const errData = await res.json().catch(() => ({}));
            showToast(`❌ Ошибка: ${errData.detail || 'неизвестная ошибка'}`, 'error');
        }
    } catch (e) {
        showToast(`❌ Ошибка сети: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
        const toast = document.getElementById('toast');
        if (toast && toast.classList.contains('show') && toast.textContent === 'Обучение...') {
            toast.classList.remove('show');
        }
    }
}

// ---------- Тост ----------
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = type;
    toast.classList.add('show');
    clearTimeout(toast._hideTimeout);
    if (duration > 0) {
        toast._hideTimeout = setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    }
}

// ---------- Статистика и факты ----------
async function loadStats() {
    try {
        const res = await fetch('/stats');
        const data = await res.json();
        document.getElementById('statNeurons').textContent = data.neurons;
        document.getElementById('statSynapses').textContent = data.synapses;
        document.getElementById('statConcepts').textContent = data.concepts;
        document.getElementById('statKB').textContent = data.knowledge_base;
        document.getElementById('statMemory').textContent = data.memory_entries || 0;
    } catch(e) { console.error('Stats error', e); }
}

function renderFacts(facts) {
    const el = document.getElementById('factsList');
    if (!el) return;
    if (!facts || !facts.length) {
        el.innerHTML = '<p class="empty-hint">Факты появятся после ответа.</p>';
        return;
    }
    let html = '';
    facts.forEach(f => {
        const scoreText = f.score ? f.score.toFixed(2) : '?';
        html += `<div class="fact-item"><div class="q">${escapeHtml(f.q)}</div><div class="a">${escapeHtml(f.a)}</div><div class="score">score: ${scoreText}</div></div>`;
    });
    el.innerHTML = html;
}

function updateButtons() {
    const hasPair = !!(lastQuestion && lastAnswer);
    learnPosBtn.style.display = hasPair ? 'inline-flex' : 'none';
    learnNegBtn.style.display = hasPair ? 'inline-flex' : 'none';
    searchToggleBtn.style.display = hasPair ? 'inline-flex' : 'none';
}

// ---------- Тогглы панелей ----------
function updateOverlay() {
    if (!overlayEl) return;
    const anyOpen = !chatListCollapsed || settingsOpen;
    overlayEl.classList.toggle('visible', anyOpen);
}

function closeMobilePanels() {
    if (!isMobile()) return;
    let changed = false;
    if (!chatListCollapsed) { chatListCollapsed = true; changed = true; }
    if (settingsOpen) { settingsOpen = false; changed = true; }
    if (changed) {
        chatList.classList.toggle('collapsed', chatListCollapsed);
        settingsPanel.classList.toggle('open', settingsOpen);
        updateOverlay();
        saveSettings();
    }
}

function toggleChatList() {
    chatListCollapsed = !chatListCollapsed;
    chatList.classList.toggle('collapsed', chatListCollapsed);
    if (isMobile() && !chatListCollapsed && settingsOpen) {
        settingsOpen = false;
        settingsPanel.classList.remove('open');
    }
    updateOverlay();
    saveSettings();
}

function toggleSettings() {
    settingsOpen = !settingsOpen;
    settingsPanel.classList.toggle('open', settingsOpen);
    if (isMobile() && settingsOpen && !chatListCollapsed) {
        chatListCollapsed = true;
        chatList.classList.add('collapsed');
    }
    updateOverlay();
    saveSettings();
}

// ---------- Интерактивный режим агента ----------
function toggleInteractiveMode() {
    interactiveMode = interactiveModeChk.checked;
    if (interactiveMode) {
        interactiveLog.textContent = 'Интерактивный режим включён.';
        startPolling();
    } else {
        interactiveLog.textContent = 'Интерактивный режим выключен.';
        stopPolling();
        currentAgentQuestion = null;
    }
    updateAgentInteractiveConfig();
    saveSettings();
}

function startPolling() {
    if (pollIntervalId) return;
    pollIntervalId = setInterval(async () => {
        if (!interactiveMode) return;
        if (currentAgentQuestion) return;
        try {
            const res = await fetch('/agent/next_question');
            const data = await res.json();
            if (data.question) {
                currentAgentQuestion = data.question;
                addMessage('🤖 Агент спрашивает: ' + data.question, 'bot', null, true, false);
                interactiveLog.textContent = 'Ожидается ваш ответ.';
            }
        } catch(e) { console.error('Polling error', e); }
    }, 5000);
}

function stopPolling() {
    if (pollIntervalId) {
        clearInterval(pollIntervalId);
        pollIntervalId = null;
    }
}

async function updateAgentInteractiveConfig() {
    const interactive = interactiveModeChk.checked;
    const timeout = parseInt(userTimeout.value) || 30;
    try {
        await fetch('/agent/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interactive_mode: interactive, user_question_timeout: timeout })
        });
    } catch(e) { console.error('Error updating agent config:', e); }
}

// ---------- Агент-исследователь ----------
async function toggleAgent() {
    const action = agentRunning ? 'stop' : 'start';
    try {
        const res = await fetch('/agent/' + action, { method: 'POST' });
        if (res.ok) {
            agentRunning = !agentRunning;
            agentToggleBtn.textContent = agentRunning ? '⏸ Остановить' : '▶️ Запустить';
            agentLog.textContent = 'Агент ' + (agentRunning ? 'запущен' : 'остановлен');
        }
    } catch(e) { console.error(e); }
}

async function updateAgentConfig() {
    const topics = document.getElementById('agentTopics').value.split(',').map(s => s.trim()).filter(Boolean);
    const interval = parseInt(document.getElementById('agentInterval').value);
    const qCount = parseInt(document.getElementById('agentQCount').value);
    // НОВЫЕ ПАРАМЕТРЫ
    const teacherThreshold = parseFloat(document.getElementById('agentTeacherThreshold').value) || 0.7;
    const useDynamicThreshold = document.getElementById('agentUseDynamicThreshold').checked;

    if (!topics.length || interval < 1 || qCount < 1) {
        showToast('⚠️ Проверьте настройки', 'error');
        return;
    }
    try {
        const res = await fetch('/agent/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topics,
                interval,
                questions_per_cycle: qCount,
                teacher_threshold: teacherThreshold,
                use_dynamic_threshold: useDynamicThreshold
            })
        });
        const data = await res.json();
        agentLog.textContent = '✅ Настройки обновлены: темы=' + data.topics.join(', ');
        showToast('✅ Настройки агента обновлены', 'success');
    } catch(e) { console.error(e); }
    saveSettings();
}

// ---------- Обучение по теме ----------
async function trainTopic() {
    const topic = document.getElementById('trainTopic').value.trim();
    if (!topic) {
        showToast('⚠️ Введите тему', 'error');
        return;
    }
    const numPairs = parseInt(document.getElementById('trainNumPairs').value) || 30;
    const temp = parseFloat(document.getElementById('trainTemp').value) || 0.7;
    const negRatio = parseFloat(document.getElementById('trainNegRatio').value) || 0.2;
    const integrate = document.getElementById('trainIntegrate').checked;
    const epochs = parseInt(document.getElementById('trainEpochs').value) || 1;

    trainLog.textContent = '⏳ Обучение...';
    showToast('⏳ Обучение по теме "' + topic + '"...', 'info', 0);
    try {
        const res = await fetch('/train_topic', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, num_pairs: numPairs, temperature: temp, negative_ratio: negRatio, integrate, epochs })
        });
        const data = await res.json();
        trainLog.textContent = data.message || 'Готово';
        showToast('✅ Обучение завершено!', 'success');
        loadStats();
    } catch(e) {
        trainLog.textContent = '❌ Ошибка';
        showToast('❌ Ошибка при обучении', 'error');
    }
}

// ---------- Обучение пары ----------
async function trainPair() {
    const q = document.getElementById('pairQuestion').value.trim();
    const a = document.getElementById('pairAnswer').value.trim();
    const epochs = parseInt(document.getElementById('pairEpochs').value) || 1;
    if (!q || !a) {
        showToast('⚠️ Введите вопрос и ответ', 'error');
        return;
    }
    pairLog.textContent = '⏳ Обучение...';
    showToast('⏳ Обучение пары...', 'info', 0);
    try {
        const res = await fetch('/train_pair', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, answer: a, epochs })
        });
        const data = await res.json();
        pairLog.textContent = '✅ Пара выучена за ' + data.epochs + ' эпох(и)';
        showToast('✅ Пара успешно выучена!', 'success');
        loadStats();
    } catch(e) {
        pairLog.textContent = '❌ Ошибка';
        showToast('❌ Ошибка при обучении пары', 'error');
    }
}

// ---------- Сон ----------
async function sleepBrain() {
    sleepLog.textContent = '⏳ Запуск сна...';
    showToast('💤 Сон...', 'info', 0);
    try {
        const res = await fetch('/sleep', { method: 'POST' });
        const data = await res.json();
        sleepLog.textContent = '✅ Сон завершен';
        showToast('✅ Сон завершён', 'success');
        loadStats();
    } catch(e) {
        sleepLog.textContent = '❌ Ошибка';
        showToast('❌ Ошибка запуска сна', 'error');
    }
}

// Экспортируем loadStats для использования из других скриптов
window.loadStats = loadStats;