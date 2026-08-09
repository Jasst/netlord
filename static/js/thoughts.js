// ============================================================
// thoughts.js – отображение мыслей (reasoning) и краткого ответа
// ============================================================

/**
 * Форматирует текст с поддержкой markdown и блоков кода.
 * Использует глобальную функцию escapeHtml (должна быть определена в app.js).
 */
function formatMessage(text) {
    // Если функция escapeHtml не определена, определяем её локально
    if (typeof escapeHtml === 'undefined') {
        window.escapeHtml = function(text) {
            const d = document.createElement('div');
            d.textContent = text;
            return d.innerHTML;
        };
    }
    // Используем глобальную функцию
    const esc = window.escapeHtml;

    // Блоки кода
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
        let escaped = esc(content);
        escaped = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        escaped = escaped.replace(/\*(.+?)\*/g, '<em>$1</em>');
        const paragraphs = escaped.split(/\n\s*\n/);
        return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    }

    let result = '';
    for (const part of parts) {
        if (part.type === 'code') {
            result += `<pre><code class="language-${part.lang}">${esc(part.code)}</code><button class="copy-btn" onclick="window.copyCode(this)">Копировать</button></pre>`;
        } else {
            result += processText(part.content);
        }
    }
    return result;
}

/**
 * Добавляет сообщение с мыслями (сворачиваемый блок) и кратким ответом.
 * @param {string} shortAnswer – краткий ответ для пользователя.
 * @param {string|null} thoughts – полные мысли (если есть и отличаются).
 * @param {Array} facts – массив фактов.
 * @param {string} sender – 'bot' или 'user'.
 * @param {boolean} isAgentOrClarifying – флаг для стиля.
 * @param {boolean} animate – посимвольная печать.
 */
function addMessageWithThoughts(shortAnswer, thoughts, facts, sender = 'bot', isAgentOrClarifying = false, animate = false) {
    // Проверка существования messagesEl
    const messagesEl = document.getElementById('messages');
    if (!messagesEl) {
        console.error('messagesEl not found');
        return;
    }

    const div = document.createElement('div');
    div.className = 'message ' + sender;
    if (isAgentOrClarifying) div.classList.add('agent-question');

    // ---- Блок мыслей (если есть) ----
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

    // ---- Контейнер для краткого ответа ----
    const answerContainer = document.createElement('div');
    answerContainer.className = 'answer-text';
    div.appendChild(answerContainer);

    // ---- Факты и время (будут добавлены после печати) ----
    const factsHtml = (facts && facts.length) ? `<div class="facts-indicator">📚 использовано фактов: ${facts.length}</div>` : '';
    const timeHtml = `<div class="time">${new Date().toLocaleTimeString()}</div>`;

    // ---- Функция сохранения в чат ----
    function saveToChat(html) {
        // Проверяем, есть ли глобальная функция saveMessageToChat
        if (typeof saveMessageToChat === 'function') {
            saveMessageToChat(html, sender, facts, isAgentOrClarifying);
        } else {
            console.warn('saveMessageToChat not defined, message not saved');
        }
    }

    // ---- Отрисовка ----
    if (!animate) {
        // Мгновенный вывод
        answerContainer.innerHTML = formatMessage(shortAnswer);
        // Добавляем факты и время
        const extra = document.createElement('div');
        extra.innerHTML = factsHtml + timeHtml;
        div.appendChild(extra);
        messagesEl.appendChild(div);
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
        // Подсветка кода
        if (typeof highlightCodeBlocks === 'function') {
            highlightCodeBlocks();
        }
        saveToChat(div.innerHTML);
    } else {
        // Посимвольная печать
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
                // Завершено – заменяем на HTML
                answerContainer.innerHTML = formatMessage(fullText);
                const extra = document.createElement('div');
                extra.innerHTML = factsHtml + timeHtml;
                div.appendChild(extra);
                requestAnimationFrame(() => {
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                });
                if (typeof highlightCodeBlocks === 'function') {
                    highlightCodeBlocks();
                }
                saveToChat(div.innerHTML);
            }
        }
        typeNext();
    }
}

// Экспортируем функцию в глобальную область (если нужно)
window.addMessageWithThoughts = addMessageWithThoughts;
window.formatMessage = formatMessage;