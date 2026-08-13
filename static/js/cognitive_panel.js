// static/js/cognitive_panel.js
// Модуль для отображения когнитивных данных (модель мира, Self-модель, планировщик)

(function() {
    'use strict';

    // Функции загрузки
    async function loadWorldModel() {
        const statsEl = document.getElementById('worldModelStats');
        const factsEl = document.getElementById('factsListWorld');
        const hypothesesEl = document.getElementById('hypothesesList');

        if (!statsEl) return;

        try {
            // Статистика
            const statsResp = await fetch('/world_model/stats');
            if (statsResp.ok) {
                const stats = await statsResp.json();
                statsEl.innerHTML =
                    `Фактов: ${stats.facts || 0}, Убеждений: ${stats.beliefs || 0}, ` +
                    `Гипотез: ${stats.hypotheses || 0}, Эпизодов: ${stats.episodes || 0}`;
            } else {
                statsEl.textContent = 'Модель мира недоступна.';
            }

            // Факты
            const factsResp = await fetch('/world_model/facts?limit=20&min_confidence=0.3');
            if (factsResp.ok) {
                const data = await factsResp.json();
                if (data.facts && data.facts.length) {
                    factsEl.innerHTML = data.facts.map(f =>
                        `<div style="border-bottom:1px solid var(--border); padding:4px 0; display:flex; justify-content:space-between;">
                            <span><strong>${escapeHtml(f.subject)}</strong> ${escapeHtml(f.relation)} <strong>${escapeHtml(f.object)}</strong></span>
                            <span style="color:${f.confidence > 0.7 ? 'var(--success)' : 'var(--warning)'};">
                                ${(f.confidence * 100).toFixed(0)}%
                            </span>
                         </div>`
                    ).join('');
                } else {
                    factsEl.innerHTML = '<p class="empty-hint">Фактов пока нет.</p>';
                }
            }

            // Гипотезы
            const hypResp = await fetch('/world_model/hypotheses');
            if (hypResp.ok) {
                const data = await hypResp.json();
                if (data.hypotheses && data.hypotheses.length) {
                    hypothesesEl.innerHTML = data.hypotheses.map(h =>
                        `<div style="border-left:2px solid ${h.verified ? (h.result === 'confirmed' ? 'var(--success)' : 'var(--danger)') : 'var(--warning)'}; padding-left:8px; margin:4px 0;">
                            <div>${escapeHtml(h.content)}</div>
                            <div style="font-size:12px; color:var(--text-muted);">
                                Уверенность: ${(h.confidence * 100).toFixed(0)}% |
                                ${h.verified ? `Проверено: ${h.result}` : 'Ожидает проверки'}
                            </div>
                         </div>`
                    ).join('');
                } else {
                    hypothesesEl.innerHTML = '<p class="empty-hint">Нет гипотез.</p>';
                }
            }
        } catch(e) {
            console.warn('Ошибка загрузки модели мира:', e);
            statsEl.textContent = 'Ошибка загрузки.';
        }
    }

    async function loadSelfModel() {
        const container = document.getElementById('selfModelContent');
        if (!container) return;

        try {
            const resp = await fetch('/self_model/status');
            if (resp.ok) {
                const data = await resp.json();
                let html = '';
                if (data.capabilities && Object.keys(data.capabilities).length) {
                    html += `<div><strong>Способности:</strong> ${Object.entries(data.capabilities).map(([k,v]) => `${k} (${(v*100).toFixed(0)}%)`).join(', ')}</div>`;
                }
                if (data.recent_errors && data.recent_errors.length) {
                    html += `<div><strong>Недавние ошибки:</strong> ${data.recent_errors.join('; ')}</div>`;
                }
                if (data.successful_strategies && data.successful_strategies.length) {
                    html += `<div><strong>Успешные стратегии:</strong> ${data.successful_strategies.join('; ')}</div>`;
                }
                if (data.unresolved_questions && data.unresolved_questions.length) {
                    html += `<div><strong>Нерешённые вопросы:</strong> ${data.unresolved_questions.join('; ')}</div>`;
                }
                if (data.knowledge && data.knowledge.length) {
                    html += `<div><strong>Знания о себе:</strong> ${data.knowledge.map(k => `${k.aspect}: ${k.description} (${(k.confidence*100).toFixed(0)}%)`).join('; ')}</div>`;
                }
                container.innerHTML = html || '<p class="empty-hint">Нет данных о себе.</p>';
            } else {
                container.innerHTML = '<p class="empty-hint">Self-модель недоступна.</p>';
            }
        } catch(e) {
            console.warn('Ошибка загрузки Self-модели:', e);
            container.innerHTML = '<p class="empty-hint">Ошибка загрузки.</p>';
        }
    }

    async function loadPlanner() {
        const container = document.getElementById('plannerContent');
        if (!container) return;

        try {
            const resp = await fetch('/planner/goals');
            if (resp.ok) {
                const data = await resp.json();
                if (data.goals && data.goals.length) {
                    container.innerHTML = data.goals.map(g =>
                        `<div style="border:1px solid var(--border); border-radius:6px; padding:8px; margin-bottom:8px;">
                            <div><strong>${escapeHtml(g.description)}</strong> <span style="color:${g.status === 'done' ? 'var(--success)' : g.status === 'failed' ? 'var(--danger)' : 'var(--warning)'};">${g.status}</span></div>
                            ${g.plan && g.plan.length ? `<div style="font-size:12px; color:var(--text-muted);">План: ${g.plan.map(a => `${a.type} (${a.status})`).join(' → ')}</div>` : ''}
                         </div>`
                    ).join('');
                } else {
                    container.innerHTML = '<p class="empty-hint">Нет активных целей.</p>';
                }
            } else {
                container.innerHTML = '<p class="empty-hint">Планировщик недоступен.</p>';
            }
        } catch(e) {
            console.warn('Ошибка загрузки планов:', e);
            container.innerHTML = '<p class="empty-hint">Ошибка загрузки.</p>';
        }
    }

    // Вспомогательная функция для безопасного вывода HTML
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Инициализация при загрузке DOM
    document.addEventListener('DOMContentLoaded', function() {
        // Загружаем данные
        loadWorldModel();
        loadSelfModel();
        loadPlanner();

        // Кнопки обновления
        const refreshWorldBtn = document.getElementById('refreshWorldModelBtn');
        if (refreshWorldBtn) refreshWorldBtn.addEventListener('click', loadWorldModel);

        const refreshSelfBtn = document.getElementById('refreshSelfModelBtn');
        if (refreshSelfBtn) refreshSelfBtn.addEventListener('click', loadSelfModel);

        const refreshPlannerBtn = document.getElementById('refreshPlannerBtn');
        if (refreshPlannerBtn) refreshPlannerBtn.addEventListener('click', loadPlanner);

        // Автообновление каждые 30 секунд
        setInterval(() => {
            loadWorldModel();
            loadSelfModel();
            loadPlanner();
        }, 30000);
    });

})();