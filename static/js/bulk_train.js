// bulk_train.js – загрузка и обучение из JSONL через интерфейс
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadJsonlForm');
    const fileInput = document.getElementById('jsonlFileInput');
    const uploadBtn = document.getElementById('uploadJsonlBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const trainLogContainer = document.getElementById('bulkTrainLog');

    if (!uploadForm) return;

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            alert('Выберите файл в формате JSONL.');
            return;
        }

        // Проверяем расширение (необязательно)
        if (!file.name.endsWith('.jsonl') && !file.name.endsWith('.json')) {
            alert('Пожалуйста, выберите файл с расширением .jsonl или .json');
            return;
        }

        // Формируем FormData
        const formData = new FormData();
        formData.append('file', file);

        // Базовые параметры
        const sleepEvery = parseInt(document.getElementById('bulkSleepEvery')?.value) || 200;
        const reward = parseFloat(document.getElementById('bulkReward')?.value) || 1.0;
        formData.append('sleep_every', sleepEvery);
        formData.append('reward', reward);

        // НОВЫЕ ПАРАМЕТРЫ ВАЛИДАЦИИ
        const validate = document.getElementById('bulkValidate')?.checked || false;
        const threshold = parseFloat(document.getElementById('bulkThreshold')?.value) || 0.6;
        const retries = parseInt(document.getElementById('bulkRetries')?.value) || 1;
        formData.append('validate', validate ? 'true' : 'false');
        formData.append('validation_threshold', threshold);
        formData.append('validation_retries', retries);

        // Блокируем кнопку
        uploadBtn.disabled = true;
        uploadBtn.textContent = '⏳ Загрузка...';
        uploadStatus.textContent = 'Загрузка файла...';

        try {
            const response = await fetch('/upload_jsonl', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Ошибка сервера');
            }

            const data = await response.json();
            uploadStatus.textContent = `✅ Файл загружен. Job ID: ${data.job_id}`;
            // Запускаем опрос логов
            pollJobLogs(data.job_id);
        } catch (err) {
            uploadStatus.textContent = `❌ Ошибка: ${err.message}`;
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '📤 Загрузить и обучить';
        }
    });

    // Функция опроса логов
    async function pollJobLogs(jobId) {
        const logDiv = trainLogContainer || document.getElementById('bulkTrainLog');
        logDiv.innerHTML = '⏳ Обучение запущено... Ждём логов...';

        let lastLine = 0;
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/upload_jsonl/logs/${jobId}?last=${lastLine}`);
                if (!res.ok) {
                    // Если джоб завершён или ошибка
                    const data = await res.json();
                    if (data.status === 'completed') {
                        logDiv.innerHTML += `\n✅ Обучение завершено! (пар: ${data.total_pairs || '?'})`;
                        clearInterval(interval);
                        return;
                    } else if (data.status === 'error') {
                        logDiv.innerHTML += `\n❌ Ошибка: ${data.error}`;
                        clearInterval(interval);
                        return;
                    }
                    return;
                }
                const data = await res.json();
                if (data.lines && data.lines.length) {
                    const newLines = data.lines.join('\n');
                    logDiv.innerHTML += '\n' + newLines;
                    logDiv.scrollTop = logDiv.scrollHeight;
                }
                lastLine = data.last_line || lastLine;
                if (data.done) {
                    clearInterval(interval);
                    logDiv.innerHTML += '\n🏁 Обучение завершено.';
                    // Перезагружаем статистику
                    if (typeof loadStats === 'function') loadStats();
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        }, 2000);

        // Таймаут на случай, если что-то пошло не так
        setTimeout(() => {
            clearInterval(interval);
        }, 3600000); // 1 час максимум
    }
});