document.addEventListener('DOMContentLoaded', function () {
    if (typeof vis === 'undefined') {
        alert('Библиотека vis-network не загружена.');
        return;
    }
    let network = null;
    let nodes = new vis.DataSet([]);
    let edges = new vis.DataSet([]);
    let selectedNode = null;
    let selectedEdge = null;
    let pendingEdgeSource = null;
    let dragSourceId = null;

    let savedPositions = {};
    let draggingNodeId = null;
    let hoverTargetId = null;
    const repelThreshold = 120;
    const repelStrength = 0.35;

    let darkMode = true;

    const container = document.getElementById('graph-container');
    const infoPanel = document.getElementById('info-panel');
    const infoContent = document.getElementById('infoContent');
    const loadingIndicator = document.getElementById('loading-indicator');
    const focusNodeBtn = document.getElementById('focusNodeBtn');
    const weightInput = document.getElementById('weightInput');
    const contextMenu = document.getElementById('contextMenu');

    const typeColors = {
        SENSORY: '#8F8178',
        CONCEPT: '#6F9587',
        MOTOR: '#7285A0',
        EMOTIONAL: '#9B7890',
        ATTENTION: '#A49A65'
    };
    const defaultColor = '#77736C';
    const selectionColor = '#E1C56E';
    const edgePositiveColor = '#43D889';
    const edgeNegativeColor = '#FF6262';
    const edgeSelectedColor = '#FFD84D';

    function getGrayColor(type) {
        return typeColors[type] || defaultColor;
    }

    function getWeight(e) {
        return Number(e.weight !== undefined ? e.weight : e.value !== undefined ? e.value : 0) || 0;
    }

    function getNodeSize(degree, base = 18) {
        return Math.max(12, Math.min(45, base + Math.log(degree + 1) * 6));
    }

    // ===== АДАПТИВНАЯ ИНФОРМАЦИОННАЯ ПАНЕЛЬ =====
    function adjustInfoPanel() {
        if (!infoPanel || !infoPanel.classList.contains('show')) return;
        const maxWidth = Math.min(320, window.innerWidth * 0.9);
        infoPanel.style.maxWidth = maxWidth + 'px';
        const rect = infoPanel.getBoundingClientRect();
        if (rect.right > window.innerWidth - 10) {
            infoPanel.style.right = '10px';
        } else {
            infoPanel.style.right = '20px';
        }
        if (window.innerWidth < 600) {
            infoPanel.style.bottom = '10px';
            infoPanel.style.right = '10px';
            infoPanel.style.maxHeight = '60vh';
        } else {
            infoPanel.style.bottom = '20px';
            infoPanel.style.maxHeight = '70vh';
        }
    }

    // Вызываем при изменении размера окна
    window.addEventListener('resize', adjustInfoPanel);

    // Инициализация сети
    function initNetwork(data) {
        const rawNodes = data.nodes || [];
        const rawEdges = data.edges || [];
        const degree = {};
        rawNodes.forEach(n => { degree[n.id] = 0; });
        rawEdges.forEach(e => {
            degree[e.from] = (degree[e.from] || 0) + 1;
            degree[e.to] = (degree[e.to] || 0) + 1;
        });
        const weights = rawEdges.map(getWeight);
        let minW = weights.length ? Math.min(...weights.map(Math.abs)) : 0;
        let maxW = weights.length ? Math.max(...weights.map(Math.abs)) : 1;
        if (minW === maxW) maxW = minW + 0.001;
        const minLen = 80;
        const maxLen = 260;

        function weightToLength(weight) {
            const w = Math.abs(weight);
            const norm = (w - minW) / (maxW - minW);
            return maxLen - norm * (maxLen - minLen);
        }

        const nodeData = rawNodes.map(n => {
            const type = n.node_type || (n.title ? n.title.split('Тип:')[1]?.split('\n')[0] : 'CONCEPT') || 'CONCEPT';
            const color = getGrayColor(type);
            const size = getNodeSize(degree[n.id] || 0);
            return {
                ...n,
                id: n.id,
                type: type,
                label: n.label || n.full_label || `#${n.id}`,
                title: n.full_label || n.label || `Нейрон #${n.id}`,
                size: size,
                originalColor: color,
                color: {
                    background: color,
                    border: color,
                    highlight: { background: selectionColor, border: '#FFF0A8' },
                    hover: { background: '#CDBB88', border: '#F4E4AA' }
                },
                borderWidth: 1.5,
                shadow: {
                    enabled: degree[n.id] < 80,
                    color: 'rgba(0,0,0,.45)',
                    size: 8,
                    x: 0,
                    y: 2
                }
            };
        });

        const edgeData = rawEdges.map(e => {
            const weight = getWeight(e);
            const positive = weight >= 0;
            const color = positive ? edgePositiveColor : edgeNegativeColor;
            const width = Math.max(1, Math.min(5, 1 + Math.abs(weight) * 3));
            return {
                ...e,
                weight: weight,
                originalColor: color,
                originalWidth: width,
                width: width,
                length: weightToLength(weight),
                color: {
                    color: color,
                    opacity: 0.65,
                    highlight: edgeSelectedColor,
                    hover: '#FFFFFF'
                },
                arrows: { to: { enabled: true, scaleFactor: 0.55 } },
                smooth: { type: 'curvedCW', roundness: 0.16 },
                shadow: { enabled: false }
            };
        });

        nodes = new vis.DataSet(nodeData);
        edges = new vis.DataSet(edgeData);

        const options = {
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -35,
                    centralGravity: 0.015,
                    springLength: 140,
                    springConstant: 0.035,
                    damping: 0.9,
                    avoidOverlap: 0.8
                },
                stabilization: {
                    enabled: true,
                    iterations: 500,
                    updateInterval: 50,
                    onlyDynamicEdges: false,
                    fit: true
                },
                maxVelocity: 10,
                minVelocity: 0.1,
                timestep: 0.5
            },
            interaction: {
                hover: true,
                tooltipDelay: 150,
                dragNodes: true,
                dragView: true,
                zoomView: true,
                navigationButtons: true,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false
            },
            nodes: {
                shape: 'dot',
                font: {
                    color: darkMode ? '#EDE6DB' : '#222',
                    size: 12,
                    face: '-apple-system, Segoe UI, Inter, sans-serif',
                    strokeWidth: 3,
                    strokeColor: darkMode ? '#211D19' : '#f5f5f5'
                },
                borderWidth: 1.5,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,.4)',
                    size: 8,
                    x: 0,
                    y: 2
                }
            },
            edges: {
                smooth: { type: 'curvedCW', roundness: 0.16 },
                arrows: { to: { enabled: true, scaleFactor: 0.55 } },
                selectionWidth: 3,
                hoverWidth: 1.5,
                font: {
                    color: '#FFF',
                    size: 10,
                    strokeWidth: 2,
                    strokeColor: '#000'
                }
            },
            layout: { improvedLayout: true, hierarchical: false }
        };

        network = new vis.Network(container, { nodes, edges }, options);
        setupEvents();
        network.startSimulation();

        network.once('stabilizationIterationsDone', function () {
            network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
            network.setOptions({ physics: { enabled: false } });
            loadingIndicator?.classList.remove('show');
        });
        setTimeout(function () {
            if (!network) return;
            network.setOptions({ physics: { enabled: false } });
            network.fit();
            loadingIndicator?.classList.remove('show');
        }, 5000);
    }

    // ---------- МАССОВЫЕ ОБНОВЛЕНИЯ ----------
    function highlightNode(id) {
        const connected = new Set([id]);
        const connectedEdges = new Set();
        edges.forEach(e => {
            if (e.from === id || e.to === id) {
                connected.add(e.from);
                connected.add(e.to);
                connectedEdges.add(e.id);
            }
        });

        const nodeUpdates = [];
        const edgeUpdates = [];

        nodes.forEach(n => {
            const active = connected.has(n.id);
            const selected = n.id === id;
            if (active) {
                nodeUpdates.push({
                    id: n.id,
                    color: {
                        background: selected ? selectionColor : n.originalColor,
                        border: selected ? '#FFF0A8' : n.originalColor
                    }
                });
            } else {
                nodeUpdates.push({
                    id: n.id,
                    color: { background: '#36322E', border: '#46413B' }
                });
            }
        });

        edges.forEach(e => {
            if (connectedEdges.has(e.id)) {
                edgeUpdates.push({
                    id: e.id,
                    color: { color: e.originalColor, opacity: 0.95, highlight: '#FFF0A0' },
                    width: Math.max(e.originalWidth, 2)
                });
            } else {
                edgeUpdates.push({
                    id: e.id,
                    color: { color: '#403B36', opacity: 0.08 },
                    width: 0.5
                });
            }
        });

        if (nodeUpdates.length) nodes.update(nodeUpdates);
        if (edgeUpdates.length) edges.update(edgeUpdates);
    }

    function restoreAppearance() {
        const nodeUpdates = [];
        const edgeUpdates = [];

        nodes.forEach(n => {
            nodeUpdates.push({
                id: n.id,
                color: {
                    background: n.originalColor || defaultColor,
                    border: n.originalColor || defaultColor
                }
            });
        });

        edges.forEach(e => {
            edgeUpdates.push({
                id: e.id,
                color: { color: e.originalColor, opacity: 0.65, highlight: edgeSelectedColor, hover: '#FFF' },
                width: e.originalWidth || 1
            });
        });

        if (nodeUpdates.length) nodes.update(nodeUpdates);
        if (edgeUpdates.length) edges.update(edgeUpdates);
    }

    function highlightEdge(id) {
        const edgeUpdates = [];
        edges.forEach(e => {
            if (e.id === id) {
                edgeUpdates.push({
                    id: e.id,
                    color: { color: edgeSelectedColor, opacity: 1, highlight: '#FFF' },
                    width: Math.max(e.originalWidth || 1, 4)
                });
            } else {
                edgeUpdates.push({
                    id: e.id,
                    color: { color: '#403B36', opacity: 0.1 },
                    width: 0.5
                });
            }
        });
        if (edgeUpdates.length) edges.update(edgeUpdates);
    }

    // ---------- ПОДСВЕТКА ЦЕЛИ ----------
    function highlightTarget(nodeId) {
        if (nodeId === null) return;
        const node = nodes.get(nodeId);
        if (!node) return;
        nodes.update({
            id: nodeId,
            color: {
                background: node.originalColor,
                border: '#FFD700'
            },
            borderWidth: 4
        });
    }

    function unhighlightTarget() {
        if (hoverTargetId === null) return;
        const node = nodes.get(hoverTargetId);
        if (node) {
            nodes.update({
                id: hoverTargetId,
                color: {
                    background: node.originalColor,
                    border: node.originalColor
                },
                borderWidth: 1.5
            });
        }
        hoverTargetId = null;
    }

    // ---------- СОЗДАНИЕ РЕБРА ----------
    function createEdge(from, to) {
        const weight = parseFloat(weightInput?.value) || 0.5;
        if (from === to) {
            alert('Нельзя соединить узел с самим собой.');
            pendingEdgeSource = null;
            return;
        }
        if (!confirm(`Создать ребро ${from} → ${to} с весом ${weight}?`)) {
            pendingEdgeSource = null;
            return;
        }
        fetch('/graph/add_edge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from, to, weight })
        })
            .then(resp => {
                if (!resp.ok) throw new Error('Ошибка создания ребра');
                pendingEdgeSource = null;
                loadGraph();
            })
            .catch(e => {
                pendingEdgeSource = null;
                alert('Ошибка сети: ' + e.message);
            });
    }

    // ---------- ИНФОРМАЦИОННЫЕ ПАНЕЛИ ----------
    function showNodeInfo(id) {
        if (!infoContent) return;
        const node = nodes.get(id);
        if (!node) return;
        const neighborEdges = edges.get().filter(e => e.from === id || e.to === id);
        const degree = neighborEdges.length;
        const avgWeight = neighborEdges.reduce((sum, e) => sum + Math.abs(getWeight(e)), 0) / (degree || 1);

        const neighborRows = neighborEdges.slice()
            .sort((a, b) => Math.abs(getWeight(b)) - Math.abs(getWeight(a)))
            .slice(0, 15)
            .map(e => {
                const otherId = e.from === id ? e.to : e.from;
                const other = nodes.get(otherId);
                const arrow = e.from === id ? '→' : '←';
                const label = other ? (other.full_label || other.label || `#${otherId}`) : `#${otherId}`;
                const w = getWeight(e);
                return `<div class="neighbor-row" data-goto="${otherId}">
                            <span class="arrow">${arrow}</span>
                            <span class="n-label">${label}</span>
                            <span class="${w < 0 ? 'w neg' : 'w'}">${w.toFixed(3)}</span>
                        </div>`;
            }).join('');

        infoContent.innerHTML = `
            <div class="info-row"><span class="info-label">ID</span><span class="info-value">${id}</span></div>
            <div class="info-row"><span class="info-label">Метка</span><span class="info-value">${node.full_label || node.label || '—'}</span></div>
            <div class="info-row"><span class="info-label">Тип</span><span class="info-value">${node.type || '—'}</span></div>
            <div class="info-row"><span class="info-label">Кластер</span><span class="info-value">${node.cluster || '—'}</span></div>
            <div class="info-row"><span class="info-label">Связей</span><span class="info-value">${degree}</span></div>
            <div class="info-row"><span class="info-label">Ср. вес связей</span><span class="info-value">${avgWeight.toFixed(3)}</span></div>
            ${neighborEdges.length ? `
                <div class="neighbors-title">Связанные понятия</div>
                <div class="neighbors">${neighborRows}</div>
                ${neighborEdges.length > 15 ? `<div class="hint">… и ещё ${neighborEdges.length - 15}</div>` : ''}
            ` : `<div class="hint">Узел пока ни с чем не связан.</div>`}
            <div style="margin-top:12px; border-top:1px solid var(--border); padding-top:10px;">
                <button id="startEdgeFromNode" style="background:var(--accent);color:#fff;border:none;padding:6px 12px;border-radius:6px;cursor:pointer;width:100%;">➕ Создать ребро от этого узла</button>
            </div>
        `;

        infoPanel.classList.add('show');
        infoPanel.style.display = 'block';
        adjustInfoPanel(); // адаптация

        const startEdgeBtn = document.getElementById('startEdgeFromNode');
        if (startEdgeBtn) {
            startEdgeBtn.onclick = function () {
                pendingEdgeSource = id;
                infoContent.innerHTML = `
                    <div style="text-align:center;padding:10px;">
                        <p>🔗 Теперь кликните на <strong>целевой узел</strong>.</p>
                        <p style="font-size:12px;color:var(--text-muted);">Вес: ${weightInput?.value || 0.5}</p>
                        <button id="cancelEdgeCreation" style="background:var(--danger-soft);color:var(--danger);border:1px solid var(--danger);padding:6px 12px;border-radius:6px;cursor:pointer;">Отменить</button>
                    </div>
                `;
                document.getElementById('cancelEdgeCreation')?.addEventListener('click', function () {
                    pendingEdgeSource = null;
                    showNodeInfo(id);
                });
                network.selectNodes([]);
            };
        }

        infoContent.querySelectorAll('.neighbor-row').forEach(row => {
            row.addEventListener('click', function () {
                const targetId = parseInt(row.dataset.goto, 10);
                if (!nodes.get(targetId)) return;
                selectedNode = targetId;
                selectedEdge = null;
                network.selectNodes([targetId]);
                network.focus(targetId, { scale: 1.3, animation: true });
                highlightNode(targetId);
                showNodeInfo(targetId);
            });
        });

        if (focusNodeBtn) {
            focusNodeBtn.onclick = function () {
                if (network && selectedNode !== null) {
                    network.focus(selectedNode, { scale: 1.5, animation: true });
                }
            };
        }
        document.getElementById('animateActivationBtn').onclick = function() {
            if (selectedNode !== null) animateActivation(selectedNode);
        };
    }

    function showEdgeInfo(id) {
        if (!infoContent) return;
        const edge = edges.get(id);
        if (!edge) return;
        const weight = getWeight(edge);
        infoContent.innerHTML = `
            <div class="info-row"><span class="info-label">Ребро</span><span class="info-value">${edge.from} → ${edge.to}</span></div>
            <div class="info-row"><span class="info-label">Вес</span><span class="info-value" style="color:${weight >= 0 ? edgePositiveColor : edgeNegativeColor};">${weight.toFixed(4)}</span></div>
            <div class="info-row"><span class="info-label">Тип</span><span class="info-value">${weight >= 0 ? 'положительная' : 'отрицательная'}</span></div>
            <div class="hint">Двойной клик по ребру — изменить вес.</div>
        `;
        infoPanel.classList.add('show');
        infoPanel.style.display = 'block';
        adjustInfoPanel(); // адаптация
    }

    // ---------- АНИМАЦИЯ АКТИВАЦИИ (с учётом силы связи) ----------
    function animateActivation(startId) {
        if (!network) return;
        const steps = 3;
        let current = [startId];
        const activated = new Set();

        function stepActivation(step) {
            if (step >= steps) {
                setTimeout(() => {
                    nodes.forEach(n => {
                        if (activated.has(n.id)) {
                            nodes.update({ id: n.id, color: { background: n.originalColor, border: n.originalColor } });
                        }
                    });
                }, 300);
                return;
            }
            const neighbors = [];
            current.forEach(id => {
                const connected = edges.get().filter(e => e.from === id || e.to === id);
                connected.forEach(e => {
                    const target = e.from === id ? e.to : e.from;
                    if (!activated.has(target)) {
                        activated.add(target);
                        neighbors.push(target);
                        // Вычисляем интенсивность на основе веса ребра
                        const weight = getWeight(e);
                        const intensity = Math.min(1, Math.abs(weight) * 2);
                        const color = `rgba(255, 215, 0, ${intensity})`;
                        nodes.update({ id: target, color: { background: color, border: color } });
                    }
                });
            });
            current = neighbors;
            setTimeout(() => stepActivation(step + 1), 500);
        }
        activated.add(startId);
        stepActivation(0);
    }

    // ---------- ЗАГРУЗКА ГРАФА ----------
    async function loadGraph() {
        loadingIndicator?.classList.add('show');
        try {
            const resp = await fetch('/graph/data?limit=500');
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(`${resp.status}\n${text}`);
            }
            const data = await resp.json();
            if (network) {
                network.destroy();
                network = null;
            }
            selectedNode = null;
            selectedEdge = null;
            pendingEdgeSource = null;
            dragSourceId = null;
            initNetwork(data);
        } catch (e) {
            console.error(e);
            alert(`Ошибка загрузки графа:\n${e.message}`);
            loadingIndicator?.classList.remove('show');
        }
    }

    // ---------- УПРАВЛЕНИЕ ФИЗИКОЙ ----------
    function resetPhysics() {
        if (!network) return;
        network.setOptions({ physics: { enabled: true } });
        network.startSimulation();
        setTimeout(function () {
            if (!network) return;
            network.setOptions({ physics: { enabled: false } });
            network.fit({ animation: { duration: 500 } });
        }, 1500);
    }

    // ---------- УДАЛЕНИЕ ----------
    async function deleteNode() {
        if (selectedNode === null) {
            alert('Сначала выберите узел.');
            return;
        }
        if (!confirm(`Удалить узел ${selectedNode} и все его связи?`)) return;
        try {
            const resp = await fetch('/graph/delete_node', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: selectedNode })
            });
            if (!resp.ok) throw new Error('Ошибка удаления');
            selectedNode = null;
            infoPanel.classList.remove('show');
            infoPanel.style.display = 'none';
            loadGraph();
        } catch (e) {
            alert('Ошибка сети: ' + e.message);
        }
    }

    async function deleteEdge() {
        if (selectedEdge === null) {
            alert('Сначала выберите ребро.');
            return;
        }
        const edge = edges.get(selectedEdge);
        if (!edge) return;
        if (!confirm(`Удалить ребро ${edge.from} → ${edge.to}?`)) return;
        try {
            const resp = await fetch('/graph/delete_edge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ from: edge.from, to: edge.to })
            });
            if (!resp.ok) throw new Error('Ошибка удаления');
            selectedEdge = null;
            infoPanel.classList.remove('show');
            infoPanel.style.display = 'none';
            loadGraph();
        } catch (e) {
            alert('Ошибка сети: ' + e.message);
        }
    }

    // ---------- ОБНОВЛЕНИЕ УЗЛА ----------
    async function updateLabel() {
        const input = document.getElementById('newLabelInput');
        const typeSelect = document.getElementById('editNodeTypeSelect');
        const reembedCheckbox = document.getElementById('reembedCheckbox');
        if (selectedNode === null) {
            alert('Сначала выберите узел.');
            return;
        }
        const newLabel = input.value.trim();
        const nodeType = typeSelect.value;
        const reembed = reembedCheckbox.checked;
        if (!newLabel && !nodeType) {
            alert('Введите новую метку и/или выберите тип.');
            return;
        }
        const body = { id: selectedNode, reembed: reembed };
        if (newLabel) body.label = newLabel;
        if (nodeType) body.node_type = nodeType;
        try {
            const resp = await fetch('/graph/update_label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!resp.ok) throw new Error('Ошибка обновления');
            input.value = '';
            typeSelect.value = '';
            loadGraph();
        } catch (e) {
            alert('Ошибка сети: ' + e.message);
        }
    }

    // ---------- РУЧНОЕ ДОБАВЛЕНИЕ РЕБРА ----------
    function addEdgeManually() {
        if (selectedNode === null) {
            alert('Сначала выберите узел-источник.');
            return;
        }
        const value = prompt('Введите ID узла-приёмника:');
        if (!value) return;
        const to = parseInt(value, 10);
        if (Number.isNaN(to)) {
            alert('ID должен быть числом.');
            return;
        }
        createEdge(selectedNode, to);
    }

    // ---------- ДОБАВЛЕНИЕ УЗЛА ----------
    async function addNode() {
        const input = document.getElementById('newNodeLabelInput');
        const typeSelect = document.getElementById('newNodeTypeSelect');
        const label = input.value.trim();
        const nodeType = typeSelect.value;
        if (!label) {
            alert('Введите название понятия.');
            return;
        }
        try {
            const resp = await fetch('/graph/add_node', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label, node_type: nodeType })
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text);
            }
            input.value = '';
            loadGraph();
        } catch (e) {
            alert('Ошибка добавления узла:\n' + e.message);
        }
    }

    // ---------- СОХРАНЕНИЕ ----------
    async function saveModel() {
        try {
            const resp = await fetch('/brain/save', { method: 'POST' });
            if (resp.ok) {
                alert('Модель сохранена.');
            } else {
                alert('Ошибка сохранения.');
            }
        } catch (e) {
            alert('Ошибка сети: ' + e.message);
        }
    }

    // ---------- ЭКСПОРТ PNG ----------
    function exportPNG() {
        if (!network) return;
        network.canvas.toBlob(function(blob) {
            const link = document.createElement('a');
            link.download = 'graph.png';
            link.href = URL.createObjectURL(blob);
            link.click();
        });
    }

    // ---------- ПОИСК УЗЛА ----------
    function searchNode() {
        const query = document.getElementById('searchNodeInput').value.trim().toLowerCase();
        if (!query) return;
        const found = nodes.get().find(n =>
            n.label?.toLowerCase().includes(query) ||
            n.full_label?.toLowerCase().includes(query)
        );
        if (found) {
            selectedNode = found.id;
            selectedEdge = null;
            network.selectNodes([found.id]);
            network.focus(found.id, { scale: 1.5, animation: true });
            highlightNode(found.id);
            showNodeInfo(found.id);
        } else {
            alert('Узел не найден');
        }
    }

    // ---------- ПЕРЕКЛЮЧЕНИЕ ТЕМЫ ----------
    function toggleTheme() {
        darkMode = !darkMode;
        document.body.classList.toggle('light', !darkMode);
        if (network) {
            network.setOptions({
                nodes: {
                    font: {
                        color: darkMode ? '#EDE6DB' : '#222',
                        strokeColor: darkMode ? '#211D19' : '#f5f5f5'
                    }
                }
            });
            network.fit();
        }
    }

    // ---------- КОНТЕКСТНОЕ МЕНЮ ----------
    function showContextMenu(x, y, nodeId) {
        contextMenu.style.left = x + 'px';
        contextMenu.style.top = y + 'px';
        contextMenu.style.display = 'block';
        contextMenu.dataset.nodeId = nodeId;
    }
    function hideContextMenu() {
        contextMenu.style.display = 'none';
    }
    document.addEventListener('click', hideContextMenu);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideContextMenu(); });

    // ---------- НАСТРОЙКА СОБЫТИЙ ГРАФА ----------
    function setupEvents() {
        network.on('hoverNode', function (params) {
            if (selectedNode === null) highlightNode(params.node);
        });
        network.on('blurNode', function () {
            if (selectedNode === null) restoreAppearance();
        });

        network.on('dragStart', function (params) {
            network.setOptions({ physics: { enabled: false } });
            if (params.nodes.length) {
                dragSourceId = params.nodes[0];
                draggingNodeId = params.nodes[0];
                savedPositions = network.getPositions();
                unhighlightTarget();
            }
        });

        network.on('dragging', function (params) {
            if (draggingNodeId !== null) {
                const dragPos = network.getPosition(draggingNodeId);
                if (dragPos) {
                    const allPositions = network.getPositions();
                    const updates = [];
                    for (let id in allPositions) {
                        if (id == draggingNodeId) continue;
                        const pos = allPositions[id];
                        const dx = pos.x - dragPos.x;
                        const dy = pos.y - dragPos.y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < repelThreshold && dist > 0.1) {
                            const force = (repelThreshold - dist) / repelThreshold * repelStrength;
                            updates.push({ id: id, x: pos.x + dx / dist * force, y: pos.y + dy / dist * force });
                        }
                    }
                    if (updates.length) nodes.update(updates);
                }
            }
            if (draggingNodeId !== null) {
                const target = network.getNodeAt({ x: params.pointer.DOM.x, y: params.pointer.DOM.y });
                if (target !== undefined && target !== null && target !== draggingNodeId) {
                    if (hoverTargetId !== target) {
                        unhighlightTarget();
                        hoverTargetId = target;
                        highlightTarget(target);
                    }
                } else {
                    if (hoverTargetId !== null) {
                        unhighlightTarget();
                    }
                }
            }
        });

        network.on('dragEnd', function (params) {
            if (draggingNodeId !== null) {
                const updates = [];
                for (let id in savedPositions) {
                    if (id == draggingNodeId) continue;
                    updates.push({ id: id, x: savedPositions[id].x, y: savedPositions[id].y });
                }
                if (updates.length) nodes.update(updates);
                savedPositions = {};
                draggingNodeId = null;
            }
            if (dragSourceId !== null && hoverTargetId !== null) {
                createEdge(dragSourceId, hoverTargetId);
            }
            unhighlightTarget();
            dragSourceId = null;
        });

        network.on('click', function (params) {
            restoreAppearance();
            if (params.nodes.length) {
                const nodeId = params.nodes[0];
                if (pendingEdgeSource !== null && pendingEdgeSource !== nodeId) {
                    createEdge(pendingEdgeSource, nodeId);
                    return;
                }
                pendingEdgeSource = null;
                selectedNode = nodeId;
                selectedEdge = null;
                network.selectNodes([nodeId]);
                network.selectEdges([]);
                highlightNode(nodeId);
                showNodeInfo(nodeId);
                return;
            }
            if (params.edges.length) {
                const edgeId = params.edges[0];
                selectedEdge = edgeId;
                selectedNode = null;
                network.selectEdges([edgeId]);
                network.selectNodes([]);
                highlightEdge(edgeId);
                showEdgeInfo(edgeId);
                return;
            }
            selectedNode = null;
            selectedEdge = null;
            pendingEdgeSource = null;
            network.selectNodes([]);
            network.selectEdges([]);
            infoPanel.classList.remove('show');
            infoPanel.style.display = 'none';
        });

        network.on('doubleClick', function (params) {
            if (params.edges.length) {
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId);
                if (!edge) return;
                const newWeight = prompt('Введите новый вес (от -1 до 1):', edge.weight);
                if (newWeight !== null) {
                    const w = parseFloat(newWeight);
                    if (!isNaN(w) && w >= -1 && w <= 1) {
                        fetch('/graph/update_edge', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ from: edge.from, to: edge.to, weight: w })
                        })
                            .then(resp => {
                                if (!resp.ok) throw new Error('Ошибка');
                                loadGraph();
                            })
                            .catch(e => alert('Ошибка: ' + e.message));
                    }
                }
            }
        });

        network.on('oncontext', function (params) {
            params.event.preventDefault();
            hideContextMenu();
            if (params.nodes.length) {
                const nodeId = params.nodes[0];
                selectedNode = nodeId;
                showContextMenu(params.event.clientX, params.event.clientY, nodeId);
            }
        });
    }

    // ---------- ПРИВЯЗКА КНОПОК ----------
    document.getElementById('refreshBtn')?.addEventListener('click', loadGraph);
    document.getElementById('addNodeBtn')?.addEventListener('click', addNode);
    document.getElementById('deleteNodeBtn')?.addEventListener('click', deleteNode);
    document.getElementById('deleteEdgeBtn')?.addEventListener('click', deleteEdge);
    document.getElementById('updateLabelBtn')?.addEventListener('click', updateLabel);
    document.getElementById('addEdgeBtn')?.addEventListener('click', addEdgeManually);
    document.getElementById('saveBtn')?.addEventListener('click', saveModel);
    document.getElementById('resetPhysicsBtn')?.addEventListener('click', resetPhysics);
    document.getElementById('exportBtn')?.addEventListener('click', exportPNG);
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
    document.getElementById('searchNodeBtn')?.addEventListener('click', searchNode);
    document.getElementById('searchNodeInput')?.addEventListener('keydown', e => { if (e.key === 'Enter') searchNode(); });
    document.getElementById('closeInfo')?.addEventListener('click', function () {
        selectedNode = null;
        selectedEdge = null;
        pendingEdgeSource = null;
        restoreAppearance();
        infoPanel.classList.remove('show');
        infoPanel.style.display = 'none';
        if (network) {
            network.selectNodes([]);
            network.selectEdges([]);
        }
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            pendingEdgeSource = null;
            if (network) {
                network.selectNodes([]);
                network.selectEdges([]);
            }
            restoreAppearance();
            hideContextMenu();
        }
    });

    // Контекстное меню – действия
    document.querySelectorAll('#contextMenu .menu-item').forEach(item => {
        item.addEventListener('click', function (e) {
            e.stopPropagation();
            const action = this.dataset.action;
            const nodeId = parseInt(contextMenu.dataset.nodeId);
            if (!nodeId) { hideContextMenu(); return; }
            selectedNode = nodeId;
            switch (action) {
                case 'focus':
                    network.focus(nodeId, { scale: 1.5, animation: true });
                    break;
                case 'edit':
                    const newLabel = prompt('Введите новую метку:', nodes.get(nodeId).label);
                    if (newLabel !== null && newLabel.trim()) {
                        document.getElementById('newLabelInput').value = newLabel;
                        updateLabel();
                    }
                    break;
                case 'addEdge':
                    addEdgeManually();
                    break;
                case 'animate':
                    animateActivation(nodeId);
                    break;
                case 'delete':
                    deleteNode();
                    break;
            }
            hideContextMenu();
        });
    });

document.getElementById('sleepBtn')?.addEventListener('click', async function() {
    if (!confirm('Запустить сон? Это может занять время.')) return;
    try {
        const resp = await fetch('/sleep', { method: 'POST' });  // было '/brain/sleep'
        if (resp.ok) {
            const data = await resp.json();
            const changes = data.changes || {};
            alert(`Сон завершён.\nУдалено нейронов: ${changes.neurons_removed || 0}\nУдалено синапсов: ${changes.synapses_removed || 0}\nУдалено понятий: ${changes.concepts_removed || 0}\nУдалено воспоминаний: ${changes.memory_removed || 0}`);
            loadGraph();
        } else {
            alert('Ошибка при запуске сна.');
        }
    } catch (e) {
        alert('Ошибка сети: ' + e.message);
    }
});

    // Фильтр по типам (легенда)
    document.querySelectorAll('#legend input[data-type]').forEach(cb => {
        cb.addEventListener('change', function () {
            const type = this.dataset.type;
            const visible = this.checked;
            nodes.get().forEach(n => {
                if (n.type === type) {
                    nodes.update({ id: n.id, hidden: !visible });
                }
            });
        });
    });

    loadGraph();
});