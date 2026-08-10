document.addEventListener('DOMContentLoaded', function() {
    if (typeof vis === 'undefined') {
        alert('Библиотека vis-network не загружена. Проверьте подключение к интернету.');
        return;
    }

    let network = null;
    let nodes = new vis.DataSet([]);
    let edges = new vis.DataSet([]);
    let selectedNode = null;
    let selectedEdge = null;
    let pendingEdgeSource = null;
    let dragSourceId = null; // для drag‑and‑drop

    const container = document.getElementById('graph-container');
    const infoPanel = document.getElementById('info-panel');
    const infoContent = document.getElementById('infoContent');
    const loadingIndicator = document.getElementById('loading-indicator');
    const focusNodeBtn = document.getElementById('focusNodeBtn');
    const weightInput = document.getElementById('weightInput');

    const typeColors = {
        'SENSORY': '#A6897B',
        'CONCEPT': '#7E9088',
        'MOTOR': '#7E8B9C',
        'EMOTIONAL': '#957E8B',
        'ATTENTION': '#96906F'
    };
    const defaultColor = '#7A756C';
    const selectionColor = '#B9A87F';
    const edgePositiveColor = '#44DD88';
    const edgeNegativeColor = '#FF6666';
    const edgeSelectedColor = '#FFD700';

    function getGrayColor(type) {
        return typeColors[type] || defaultColor;
    }

    function getNodeSize(degree, base = 18) {
        return Math.max(12, base + Math.log(degree + 1) * 6);
    }

    function initNetwork(data) {
        const degMap = {};
        (data.edges || []).forEach(e => {
            degMap[e.from] = (degMap[e.from] || 0) + 1;
            degMap[e.to] = (degMap[e.to] || 0) + 1;
        });

        const nodeData = (data.nodes || []).map(n => {
            const type = n.node_type || (n.title ? n.title.split('Тип:')[1]?.split('\n')[0] : 'CONCEPT');
            const degree = degMap[n.id] || 0;
            return {
                ...n,
                color: getGrayColor(type),
                originalColor: getGrayColor(type),
                size: getNodeSize(degree, 20),
                type: type
            };
        });

        nodes = new vis.DataSet(nodeData);
        edges = new vis.DataSet((data.edges || []).map(e => {
            const weight = e.weight || e.value || 0;
            return {
                ...e,
                color: weight >= 0 ? edgePositiveColor : edgeNegativeColor,
                width: Math.max(1, Math.min(4, Math.abs(weight) * 3)),
                smooth: { type: 'curvedCW', roundness: 0.15 },
                originalColor: weight >= 0 ? edgePositiveColor : edgeNegativeColor,
                originalWidth: Math.max(1, Math.min(4, Math.abs(weight) * 3))
            };
        }));

        const options = {
            physics: {
                enabled: true,
                stabilization: { iterations: 300, updateInterval: 50 },
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 120,
                    springConstant: 0.06,
                    damping: 0.9
                },
                maxVelocity: 30,
                minVelocity: 0.1
            },
            interaction: {
                hover: true,
                tooltipDelay: 300,
                navigationButtons: true,
                dragNodes: true,
                dragView: true,
                zoomView: true
            },
            nodes: {
                shape: 'dot',
                font: {
                    color: '#EDE6DB',
                    size: 12,
                    face: '-apple-system, Segoe UI, Inter, sans-serif',
                    strokeWidth: 3,
                    strokeColor: '#211D19cc'
                },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.45)',
                    size: 10,
                    x: 0,
                    y: 2
                }
            },
            edges: {
                smooth: { type: 'curvedCW', roundness: 0.2 },
                arrows: { to: { enabled: true, scaleFactor: 0.6 } },
                font: { color: '#FFFFFF', size: 11, strokeWidth: 2, strokeColor: '#000000' },
                selectionWidth: 3
            },
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        };

        network = new vis.Network(container, { nodes, edges }, options);
        network.startSimulation();

        // ========== DRAG‑AND‑DROP ДЛЯ СОЗДАНИЯ РЁБЕР ==========
        network.on('dragStart', function(params) {
            if (params.nodes.length > 0) {
                dragSourceId = params.nodes[0];
                console.log('dragStart source:', dragSourceId);
            }
        });

        network.on('dragEnd', function(params) {
            if (dragSourceId === null) return;
            const event = params.event;
            const containerRect = container.getBoundingClientRect();
            const x = event.clientX - containerRect.left;
            const y = event.clientY - containerRect.top;
            const targetNodeId = network.getNodeAt({ x, y });
            if (targetNodeId !== undefined && targetNodeId !== null && targetNodeId !== dragSourceId) {
                const weight = parseFloat(weightInput.value) || 0.5;
                if (confirm(`Создать ребро от узла ${dragSourceId} к узлу ${targetNodeId} с весом ${weight}?`)) {
                    fetch('/graph/add_edge', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ from: dragSourceId, to: targetNodeId, weight })
                    })
                    .then(resp => {
                        if (resp.ok) {
                            loadGraph();
                        } else {
                            alert('Ошибка создания ребра');
                        }
                    })
                    .catch(e => alert('Ошибка сети: ' + e.message));
                }
            }
            dragSourceId = null;
        });

        // ========== ОБРАБОТЧИК КЛИКА ==========
        network.on('click', function(params) {
            if (selectedNode !== null) {
                const prev = nodes.get(selectedNode);
                if (prev) {
                    nodes.update({ id: selectedNode, color: prev.originalColor || defaultColor });
                }
            }

            if (selectedEdge !== null) {
                const prevEdge = edges.get(selectedEdge);
                if (prevEdge) {
                    edges.update({
                        id: selectedEdge,
                        color: prevEdge.originalColor || edgePositiveColor,
                        width: prevEdge.originalWidth || 2
                    });
                }
                selectedEdge = null;
            }

            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];

                if (pendingEdgeSource !== null && pendingEdgeSource !== nodeId) {
                    const weight = parseFloat(weightInput.value) || 0.5;
                    if (confirm(`Создать ребро от узла ${pendingEdgeSource} к узлу ${nodeId} с весом ${weight}?`)) {
                        fetch('/graph/add_edge', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ from: pendingEdgeSource, to: nodeId, weight })
                        })
                        .then(resp => {
                            if (resp.ok) {
                                pendingEdgeSource = null;
                                loadGraph();
                            } else {
                                alert('Ошибка создания ребра');
                            }
                        })
                        .catch(e => alert('Ошибка сети: ' + e.message));
                        return;
                    } else {
                        pendingEdgeSource = null;
                        selectedNode = nodeId;
                        nodes.update({ id: nodeId, color: selectionColor });
                        network.selectNodes([nodeId]);
                        network.selectEdges([]);
                        showNodeInfo(nodeId);
                        return;
                    }
                }

                pendingEdgeSource = null;
                selectedNode = nodeId;
                network.selectNodes([nodeId]);
                network.selectEdges([]);
                const node = nodes.get(nodeId);
                if (node) {
                    nodes.update({ id: nodeId, color: selectionColor });
                    showNodeInfo(nodeId);
                }
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                selectedEdge = edgeId;
                network.selectEdges([edgeId]);
                network.selectNodes([]);
                const edge = edges.get(edgeId);
                if (edge) {
                    edges.update({
                        id: edgeId,
                        color: edgeSelectedColor,
                        width: Math.max(edge.originalWidth || 2, 4)
                    });
                    showEdgeInfo(edgeId);
                }
            } else {
                selectedNode = null;
                selectedEdge = null;
                pendingEdgeSource = null;
                network.selectNodes([]);
                network.selectEdges([]);
                infoPanel.classList.remove('show');
                infoPanel.style.display = 'none';
            }
        });

        network.on('hoverNode', function(params) {
            // можно добавить всплывающую подсказку, если нужно
        });

        network.on('dragStart', function() {
            pendingEdgeSource = null; // сброс при перетаскивании
        });

        network.once('stabilizationIterationsDone', function() {
            network.fit();
            loadingIndicator.classList.remove('show');
        });

        setTimeout(() => {
            network.fit();
            loadingIndicator.classList.remove('show');
        }, 5000);
    }

    // ========== ПОКАЗАТЬ ИНФОРМАЦИЮ О УЗЛЕ ==========
    function showNodeInfo(id) {
        if (!infoContent) return;
        const node = nodes.get(id);
        if (!node) return;

        const neighborEdges = edges.get().filter(e => e.from === id || e.to === id);
        const neighborRows = neighborEdges
            .slice()
            .sort((a, b) => Math.abs(b.weight ?? b.value ?? 0) - Math.abs(a.weight ?? a.value ?? 0))
            .slice(0, 15)
            .map(e => {
                const otherId = e.from === id ? e.to : e.from;
                const other = nodes.get(otherId);
                const arrow = e.from === id ? '→' : '←';
                const label = other ? (other.full_label || other.label || `#${otherId}`) : `#${otherId}`;
                const w = (e.weight !== undefined) ? e.weight.toFixed(3) : (e.value ?? '?');
                const wClass = (e.weight ?? 0) < 0 ? 'w neg' : 'w';
                return `<div class="neighbor-row" data-goto="${otherId}"><span class="arrow">${arrow}</span><span class="n-label">${label}</span><span class="${wClass}">${w}</span></div>`;
            }).join('');

        infoContent.innerHTML = `
            <div class="info-row"><span class="info-label">ID</span><span class="info-value">${node.id}</span></div>
            <div class="info-row"><span class="info-label">Метка</span><span class="info-value">${node.full_label || node.label || '—'}</span></div>
            <div class="info-row"><span class="info-label">Тип</span><span class="info-value">${node.type || '—'}</span></div>
            <div class="info-row"><span class="info-label">Кластер</span><span class="info-value">${node.cluster || '—'}</span></div>
            <div class="info-row"><span class="info-label">Связей</span><span class="info-value">${neighborEdges.length}</span></div>
            ${neighborEdges.length ? `
                <div class="neighbors-title">Связанные понятия</div>
                <div class="neighbors">${neighborRows}</div>
                ${neighborEdges.length > 15 ? `<div class="hint">… и ещё ${neighborEdges.length - 15}</div>` : ''}
            ` : `<div class="hint">Узел пока ни с чем не связан.</div>`}
            <div style="margin-top:12px; border-top:1px solid var(--border); padding-top:10px;">
                <button id="startEdgeFromNode" style="background:var(--accent); color:#fff; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; width:100%;">
                    ➕ Создать ребро от этого узла
                </button>
            </div>
        `;

        infoPanel.classList.add('show');
        infoPanel.style.display = 'block';

        // Обработчик "Создать ребро от этого узла"
        const startEdgeBtn = document.getElementById('startEdgeFromNode');
        if (startEdgeBtn) {
            startEdgeBtn.onclick = function() {
                if (!network) return;
                pendingEdgeSource = id;
                infoContent.innerHTML = `
                    <div style="text-align:center; padding:10px;">
                        <p>🔗 Теперь кликните на <strong>целевой узел</strong>, чтобы создать ребро.</p>
                        <p style="font-size:12px; color:var(--text-muted);">Вес: ${weightInput.value}</p>
                        <button id="cancelEdgeCreation" style="background:var(--danger-soft); color:var(--danger); border:1px solid var(--danger); padding:6px 12px; border-radius:6px; cursor:pointer;">Отменить</button>
                    </div>
                `;
                document.getElementById('cancelEdgeCreation').onclick = function() {
                    pendingEdgeSource = null;
                    showNodeInfo(id);
                };
                network.selectNodes([]);
            };
        }

        // Клик по соседям
        const neighborElements = infoContent.querySelectorAll('.neighbor-row');
        neighborElements.forEach(row => {
            row.addEventListener('click', function() {
                const targetId = parseInt(row.getAttribute('data-goto'), 10);
                if (!network || !nodes.get(targetId)) return;
                network.selectNodes([targetId]);
                network.focus(targetId, { scale: 1.3, animation: true });
                if (selectedNode !== null && selectedNode !== targetId) {
                    const prev = nodes.get(selectedNode);
                    if (prev) nodes.update({ id: selectedNode, color: prev.originalColor || defaultColor });
                }
                nodes.update({ id: targetId, color: selectionColor });
                selectedNode = targetId;
                selectedEdge = null;
                showNodeInfo(targetId);
            });
        });

        // Кнопка центрирования
        if (focusNodeBtn) {
            focusNodeBtn.onclick = function() {
                if (network && selectedNode !== null) {
                    network.focus(selectedNode, { scale: 1.5, animation: true });
                }
            };
        }
    }

    // ========== ПОКАЗАТЬ ИНФОРМАЦИЮ О РЕБРЕ ==========
    function showEdgeInfo(id) {
        if (!infoContent) return;
        const edge = edges.get(id);
        if (!edge) return;
        const weight = edge.weight || edge.value || 0;
        const sign = weight >= 0 ? 'положительная' : 'отрицательная';
        infoContent.innerHTML = `
            <div class="info-row"><span class="info-label">Ребро</span><span class="info-value">${edge.from} → ${edge.to}</span></div>
            <div class="info-row"><span class="info-label">Вес</span><span class="info-value">${weight}</span></div>
            <div class="info-row"><span class="info-label">Тип</span><span class="info-value">${sign}</span></div>
            <div class="hint">Для удаления нажмите «🗑 Ребро»</div>
        `;
        infoPanel.classList.add('show');
        infoPanel.style.display = 'block';
    }

    // ========== ЗАГРУЗКА ГРАФА ==========
    async function loadGraph() {
        loadingIndicator.classList.add('show');
        try {
            const resp = await fetch('/graph/data?limit=500');
            if (!resp.ok) {
                const text = await resp.text();
                alert(`Ошибка сервера: ${resp.status}\n${text}`);
                loadingIndicator.classList.remove('show');
                return;
            }
            const data = await resp.json();
            console.log('Данные графа загружены:', data);
            if (network) {
                network.destroy();
                network = null;
            }
            initNetwork(data);
            selectedNode = null;
            selectedEdge = null;
            pendingEdgeSource = null;
            dragSourceId = null;
        } catch (e) {
            alert(`Ошибка загрузки графа: ${e.message}`);
            console.error(e);
            loadingIndicator.classList.remove('show');
        }
    }

    function resetPhysics() {
        if (network) {
            network.startSimulation();
            setTimeout(() => network.fit(), 500);
        }
    }

    // ========== УДАЛЕНИЕ УЗЛА ==========
    async function deleteNode() {
        if (selectedNode === null) {
            alert('Сначала выберите узел (кликните на него)');
            return;
        }
        if (!confirm(`Удалить узел ${selectedNode} и все его связи?`)) return;
        try {
            const resp = await fetch('/graph/delete_node', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: selectedNode })
            });
            if (resp.ok) {
                selectedNode = null;
                infoPanel.classList.remove('show');
                infoPanel.style.display = 'none';
                loadGraph();
            } else {
                alert('Ошибка удаления');
            }
        } catch (e) {
            alert('Ошибка сети при удалении');
            console.error(e);
        }
    }

    // ========== УДАЛЕНИЕ РЕБРА ==========
    async function deleteEdge() {
        if (selectedEdge === null) {
            alert('Сначала выберите ребро (кликните на него)');
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
            if (resp.ok) {
                selectedEdge = null;
                infoPanel.classList.remove('show');
                infoPanel.style.display = 'none';
                loadGraph();
            } else {
                alert('Ошибка удаления');
            }
        } catch (e) {
            alert('Ошибка сети при удалении');
            console.error(e);
        }
    }

    // ========== ОБНОВЛЕНИЕ МЕТКИ / ТИПА ==========
    async function updateLabel() {
        const input = document.getElementById('newLabelInput');
        const newLabel = input.value.trim();
        const nodeType = document.getElementById('editNodeTypeSelect').value;
        const reembed = document.getElementById('reembedCheckbox').checked;
        if (selectedNode === null) {
            alert('Сначала выберите узел');
            return;
        }
        if (!newLabel && !nodeType) {
            alert('Введите новую метку и/или выберите тип');
            return;
        }
        const body = { id: selectedNode, reembed };
        if (newLabel) body.label = newLabel;
        if (nodeType) body.node_type = nodeType;
        try {
            const resp = await fetch('/graph/update_label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (resp.ok) {
                input.value = '';
                document.getElementById('editNodeTypeSelect').value = '';
                loadGraph();
            } else {
                alert('Ошибка обновления');
            }
        } catch (e) {
            alert('Ошибка сети');
            console.error(e);
        }
    }

    // ========== ДОБАВИТЬ РЕБРО (вручную) ==========
    async function addEdge() {
        if (selectedNode === null) {
            alert('Сначала выберите узел, от которого будет ребро (кликните на него)');
            return;
        }
        const toId = prompt('Введите ID узла-приёмника:');
        if (!toId) return;
        const weight = parseFloat(weightInput.value) || 0.5;
        try {
            const resp = await fetch('/graph/add_edge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ from: selectedNode, to: parseInt(toId), weight })
            });
            if (resp.ok) {
                loadGraph();
            } else {
                alert('Ошибка добавления ребра');
            }
        } catch (e) {
            alert('Ошибка сети');
            console.error(e);
        }
    }

    // ========== ДОБАВИТЬ УЗЕЛ ==========
    async function addNode() {
        const input = document.getElementById('newNodeLabelInput');
        const label = input.value.trim();
        const nodeType = document.getElementById('newNodeTypeSelect').value;
        if (!label) {
            alert('Введите название понятия (метку узла)');
            return;
        }
        try {
            const resp = await fetch('/graph/add_node', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label, node_type: nodeType })
            });
            if (resp.ok) {
                input.value = '';
                loadGraph();
            } else {
                const text = await resp.text();
                alert(`Ошибка добавления узла: ${text}`);
            }
        } catch (e) {
            alert('Ошибка сети при добавлении узла');
            console.error(e);
        }
    }

    // ========== СОХРАНИТЬ МОДЕЛЬ ==========
    async function saveModel() {
        try {
            const resp = await fetch('/brain/save', { method: 'POST' });
            if (resp.ok) alert('Модель сохранена');
            else alert('Ошибка сохранения');
        } catch (e) {
            alert('Ошибка сети');
            console.error(e);
        }
    }

    // ========== НАЗНАЧЕНИЕ КНОПОК ==========
    document.getElementById('refreshBtn').onclick = loadGraph;
    document.getElementById('addNodeBtn').onclick = addNode;
    document.getElementById('deleteNodeBtn').onclick = deleteNode;
    document.getElementById('deleteEdgeBtn').onclick = deleteEdge;
    document.getElementById('updateLabelBtn').onclick = updateLabel;
    document.getElementById('addEdgeBtn').onclick = addEdge;
    document.getElementById('saveBtn').onclick = saveModel;
    document.getElementById('resetPhysicsBtn').onclick = resetPhysics;
    document.getElementById('closeInfo').onclick = function() {
        infoPanel.classList.remove('show');
        infoPanel.style.display = 'none';
        pendingEdgeSource = null;
    };

    // ========== СТАРТ ==========
    loadGraph();
});