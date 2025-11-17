class PhotoClusterApp {
    constructor() {
        this.currentPath = '';
        // Корневая папка для обработки очереди
        this.initialPath = '';
        this.queue = [];
        this.lastTasksStr = '';
        this.pendingMoves = new Set();
        
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadInitialData();
        this.updateTasks(); // Загружаем начальный статус задач
        // Автообновление отключено по умолчанию
    }

    initializeElements() {
        this.driveButtons = document.getElementById('driveButtons');
        this.currentPathEl = document.getElementById('currentPath');
        this.folderContents = document.getElementById('folderContents');
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.queueList = document.getElementById('queueList');
        this.processBtn = document.getElementById('processBtn');
        this.processGlobalBtn = document.getElementById('processGlobalBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.includeExcludedBtn = document.getElementById('includeExcludedBtn');
        this.includeExcluded = false;
        this.jointModeSelect = document.getElementById('jointModeSelect');
        this.jointMode = 'copy'; // 'copy' or 'combine'
        this.postValidateCheckbox = document.getElementById('postValidateCheckbox');
        this.postValidate = false;
        this.addQueueBtn = document.getElementById('addQueueBtn');
        this.tasksList = document.getElementById('tasksList');
        this.clearTasksBtn = document.getElementById('clearTasksBtn');
        this.zipBtn = document.getElementById('zipBtn');

        // Элементы обновления
        this.autoRefreshBtn = document.getElementById('autoRefreshBtn');

        // Элементы управления файлами
        this.fileToolbar = document.getElementById('fileToolbar');
        this.newFolderBtn = document.getElementById('newFolderBtn');
        this.contextMenu = document.getElementById('contextMenu');
        this.createFolderModal = document.getElementById('createFolderModal');
        this.renameModal = document.getElementById('renameModal');
        this.folderNameInput = document.getElementById('folderNameInput');
        this.renameInput = document.getElementById('renameInput');
        
        // Переменные для контекстного меню
        this.contextMenuItem = null;
        this.contextItemPath = null;
        
        // Проверяем что все элементы найдены
        const elements = {
            driveButtons: this.driveButtons,
            currentPathEl: this.currentPathEl,
            folderContents: this.folderContents,
            uploadZone: this.uploadZone,
            fileInput: this.fileInput,
            queueList: this.queueList,
            processBtn: this.processBtn,
            processGlobalBtn: this.processGlobalBtn,
            clearBtn: this.clearBtn,
            addQueueBtn: this.addQueueBtn,
            tasksList: this.tasksList,
            clearTasksBtn: this.clearTasksBtn,
            zipBtn: this.zipBtn,
            jointModeSelect: this.jointModeSelect,
            postValidateCheckbox: this.postValidateCheckbox,
            refreshBtn: this.autoRefreshBtn,
            fileToolbar: this.fileToolbar,
            contextMenu: this.contextMenu
        };
        
        for (const [name, element] of Object.entries(elements)) {
            if (!element) {
                console.error(`Element not found: ${name}`);
            }
        }

        // Инициализируем селект режима
        this.jointModeSelect.value = this.jointMode;
        this.postValidateCheckbox.checked = this.postValidate;
    }

    setupEventListeners() {
        // Разрешить drop в очередь
        this.queueList.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.queueList.classList.add('drag-over');
        });
        this.queueList.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
        });
        this.queueList.addEventListener('drop', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
            const path = e.dataTransfer.getData('text/plain');
            if (path) this.addToQueue(path);
        });
        // Режим обработки совместных фото
        this.jointModeSelect.addEventListener('change', (e) => {
            this.jointMode = e.target.value;
            console.log('🔧 Joint mode changed to:', this.jointMode);
        });

        // Пост-валидация
        this.postValidateCheckbox.addEventListener('change', (e) => {
            this.postValidate = e.target.checked;
            console.log('🔧 Post validate changed to:', this.postValidate);
        });

        // Кнопки обработки очереди
        this.processBtn.addEventListener('click', () => this.processQueue());
        this.processGlobalBtn.addEventListener('click', () => this.processGlobalQueue());
        this.clearBtn.addEventListener('click', () => this.clearQueue());
        this.zipBtn.addEventListener('click', () => this.downloadZip());

        // Кнопка обновления
        this.autoRefreshBtn.addEventListener('click', () => this.manualRefresh());

        // Кнопки управления задачами
        this.clearTasksBtn.addEventListener('click', () => this.clearCompletedTasks());

        // Кнопки управления файлами
        this.newFolderBtn.addEventListener('click', () => this.openCreateFolderModal());
        
        // Контекстное меню
        this.contextMenu.addEventListener('click', (e) => {
            const action = e.target.closest('.context-menu-item')?.dataset.action;
            if (action) {
                this.handleContextAction(action);
                this.hideContextMenu();
            }
        });
        
        // Закрыть контекстное меню при клике вне его
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.context-menu')) {
                this.hideContextMenu();
            }
        });
        
        // Закрыть модальное окно при клике на фон
        [this.createFolderModal, this.renameModal].forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal.id);
                }
            });
        });
        
        // Enter для модальных окон
        this.folderNameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.createFolder();
        });
        this.renameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.renameItem();
        });
        
        // Кнопки модальных окон
        document.getElementById('cancelCreateFolderBtn').addEventListener('click', () => {
            this.closeModal('createFolderModal');
        });
        document.getElementById('confirmCreateFolderBtn').addEventListener('click', () => {
            this.createFolder();
        });
        document.getElementById('cancelRenameBtn').addEventListener('click', () => {
            this.closeModal('renameModal');
        });
        document.getElementById('confirmRenameBtn').addEventListener('click', () => {
            this.renameItem();
        });
        this.includeExcludedBtn.addEventListener('click', async () => {
            // Кнопка "Общие" - специальный алгоритм для общих фотографий
            console.log('🔍 Кнопка "Общие" нажата - запускаем специальный алгоритм для общих фото');
            
            try {
                await this.processCommonPhotosAlgorithm();
            } catch (error) {
                console.error('❌ Ошибка при обработке общих фото:', error);
                this.showNotification('Ошибка при обработке общих фото: ' + error.message, 'error');
            }
        });
        // Кнопка добавить в очередь
        this.addQueueBtn.addEventListener('click', () => this.addToQueue(this.currentPath));
        // Кнопка очистки завершенных задач
        this.clearTasksBtn.addEventListener('click', () => this.clearCompletedTasks());

        // Загрузка файлов
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));

        // Drag & Drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('drag-over');
        });

        this.uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
            this.handleFileUpload(e.dataTransfer.files);
        });
    }

    async loadInitialData() {
        await this.loadDrives();
        await this.loadQueue();
    }

    async loadDrives() {
        try {
            const response = await fetch('/api/drives', { cache: 'no-store' });
            const data = await response.json();
            const drives = data.drives || data; // Поддержка обеих форматов
            
            this.driveButtons.innerHTML = '';
            drives.forEach(drive => {
                const button = document.createElement('button');
                button.className = 'drive-btn';
                button.textContent = drive.name;
                button.addEventListener('click', () => this.navigateToFolder(drive.path));
                this.driveButtons.appendChild(button);
            });
        } catch (error) {
            this.showNotification('Ошибка загрузки дисков: ' + error.message, 'error');
        }
    }

    async navigateToFolder(path) {
        try {
            this.currentPath = path;
            // Сохраняем корневую директорию только один раз
            if (!this.initialPath) {
                this.initialPath = path;
            }
            const response = await fetch(`/api/folder?path=${encodeURIComponent(path)}&_ts=${Date.now()}`, { cache: 'no-store' });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.currentPathEl.innerHTML = `<strong>Текущая папка:</strong> ${path}`;
            // Поддержка случаев без contents
            const contents = data.contents || [];
            if (!data.contents) {
                // Формируем из папок и изображений
                if (data.folders) contents.push(...data.folders.map(f=>({name:f.name,path:f.path,is_directory:true})));
                if (data.images) contents.push(...data.images.map(i=>({name:i.name,path:i.path,is_directory:false})));
            }
            await this.displayFolderContents(contents);
            
            
            // Активируем кнопку ZIP и показываем панель инструментов
            this.zipBtn.disabled = false;
            this.fileToolbar.style.display = 'flex';
        } catch (error) {
            this.showNotification('Ошибка доступа к папке: ' + error.message, 'error');
        }
    }

    async loadFolderContents(path) {
        if (!path) {
            return;
        }

        try {
            const response = await fetch(`/api/folder?path=${encodeURIComponent(path)}&_ts=${Date.now()}`, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.path) {
                this.currentPathEl.innerHTML = `<strong>Текущая папка:</strong> ${data.path}`;
            }
            const contents = data.contents || [];
            if (!data.contents) {
                if (data.folders) contents.push(...data.folders.map(f => ({ name: f.name, path: f.path, is_directory: true })));
                if (data.images) contents.push(...data.images.map(i => ({ name: i.name, path: i.path, is_directory: false })));
            }
            await this.displayFolderContents(contents);
        } catch (error) {
            console.error('Error loading folder contents:', error);
        }
    }

    async displayFolderContents(contents) {
        this.folderContents.innerHTML = '';
        
        if (contents.length === 0) {
            this.folderContents.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Папка пуста
                </p>
            `;
            return;
        }

        for (const item of contents) {
            // Навигационная кнопка Назад
            if (item.name.includes('⬅️')) {
                const button = document.createElement('button');
                button.className = 'folder-btn back';
                button.setAttribute('draggable', 'true');
                button.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                button.textContent = item.name;
                if (item.is_directory) button.addEventListener('click', () => this.navigateToFolder(item.path));
                this.folderContents.appendChild(button);
                continue;
            }
            if (item.is_directory) {
                // Папка: если есть изображения, показываем превью, иначе кнопка
                let imgs = [];
                try {
                    const res = await fetch(`/api/folder?path=${encodeURIComponent(item.path)}&_ts=${Date.now()}`, { cache: 'no-store' });
                    const folderData = await res.json();
                    imgs = folderData.contents.filter(c => !c.is_directory);
                } catch {}
                if (imgs.length > 0) {
                    // Превью папки
                    const div = document.createElement('div');
                    div.className = 'thumbnail';
                    div.setAttribute('draggable','true');
                    div.addEventListener('click', () => this.navigateToFolder(item.path));
                    
                    // Drag & Drop для папки
                    div.addEventListener('dragstart', e => {
                        console.log('🔧 Drag start:', item.path);
                        e.dataTransfer.setData('text/plain', item.path);
                        e.dataTransfer.effectAllowed = 'move';
                    });
                    div.addEventListener('dragover', e => {
                        e.preventDefault();
                        div.classList.add('drag-over');
                    });
                    div.addEventListener('dragleave', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                    });
                    div.addEventListener('drop', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                        const src = e.dataTransfer.getData('text/plain');
                        console.log('🔧 Drop event:', src, '→', item.path);
                        this.moveItem(src, item.path);
                    });
                    
                    const img = document.createElement('img');
                    img.src = `/api/image/preview?path=${encodeURIComponent(imgs[0].path)}&size=150&_ts=${Date.now()}`;
                    img.alt = item.name.replace('📂 ', '');
                    div.appendChild(img);
                    
                    // Добавляем подпись с названием папки
                    const caption = document.createElement('div');
                    caption.className = 'thumbnail-caption';
                    caption.textContent = item.name.replace('📂 ', '');
                    div.appendChild(caption);
                    
                    // Добавляем контекстное меню
                    this.addContextMenuToElement(div, item.path, item.name);
                    
                    this.folderContents.appendChild(div);
                } else {
                    // Обычная папка без превью
                    const button = document.createElement('button');
                    button.className = 'folder-btn';
                    
                    // Проверяем, является ли папка исключаемой
                    const folderName = item.name.replace('📂 ', '');
                    const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
                    const folderNameLower = folderName.toLowerCase();
                    
                    let isExcluded = false;
                    let excludedName = '';
                    for (const name of excludedNames) {
                        if (folderNameLower.includes(name)) {
                            isExcluded = true;
                            excludedName = name;
                            break;
                        }
                    }
                    
                    if (isExcluded) {
                        button.className += ' disabled';
                        button.textContent = folderName + ' (не обрабатывается)';
                        button.title = `Папки с названием "${excludedName}" не обрабатываются`;
                        button.disabled = true;
                    } else {
                        button.textContent = folderName;
                        button.addEventListener('click', () => this.navigateToFolder(item.path));
                        
                        // Drag & Drop для обычной папки
                        button.setAttribute('draggable', 'true');
                        button.addEventListener('dragstart', e => {
                            e.dataTransfer.setData('text/plain', item.path);
                            e.dataTransfer.effectAllowed = 'move';
                        });
                        button.addEventListener('dragover', e => {
                            e.preventDefault();
                            button.classList.add('drag-over');
                        });
                        button.addEventListener('dragleave', e => {
                            e.preventDefault();
                            button.classList.remove('drag-over');
                        });
                        button.addEventListener('drop', e => {
                            e.preventDefault();
                            button.classList.remove('drag-over');
                            const src = e.dataTransfer.getData('text/plain');
                            this.moveItem(src, item.path);
                        });
                    }
                    
                    // Добавляем контекстное меню
                    if (!isExcluded) {
                        this.addContextMenuToElement(button, item.path, item.name);
                    }
                    
                    this.folderContents.appendChild(button);
                }
                continue;
            }
            // Изображение файла
            if (!item.is_directory && item.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i)) {
                const div = document.createElement('div');
                div.className = 'thumbnail';
                div.setAttribute('draggable', 'true');
                
                // Drag & Drop для изображения
                div.addEventListener('dragstart', e => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                div.addEventListener('dragover', e => {
                    e.preventDefault();
                    div.classList.add('drag-over');
                });
                div.addEventListener('dragleave', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                });
                div.addEventListener('drop', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                    const src = e.dataTransfer.getData('text/plain');
                    this.moveItem(src, item.path);
                });
                
                const img = document.createElement('img');
                img.src = `/api/image/preview?path=${encodeURIComponent(item.path)}&size=150&_ts=${Date.now()}`;
                img.alt = item.name.replace('🖼 ', '');
                div.appendChild(img);
                
                // Добавляем подпись с названием файла
                const caption = document.createElement('div');
                caption.className = 'thumbnail-caption';
                caption.textContent = item.name.replace('🖼 ', '');
                div.appendChild(caption);
                
                // Добавляем контекстное меню
                this.addContextMenuToElement(div, item.path, item.name);
                
                this.folderContents.appendChild(div);
                continue;
            }
            // Другие файлы: просто кнопка
            const button = document.createElement('button');
            button.className = 'folder-btn';
            button.textContent = item.name;
            this.folderContents.appendChild(button);
        }

        // Добавляем кнопку "Добавить в очередь" если это не навигационная кнопка
        if (!contents.some(item => item.name.includes('⬅️'))) {
            const addButton = document.createElement('button');
            addButton.className = 'action-btn';
            addButton.style.marginTop = '15px';
            addButton.textContent = '📌 Добавить в очередь';
            addButton.addEventListener('click', () => this.addToQueue(this.currentPath));
            this.folderContents.appendChild(addButton);
        }
    }

    formatFileSize(bytes) {
        const sizes = ['Б', 'КБ', 'МБ', 'ГБ'];
        if (bytes === 0) return '0 Б';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    async handleFileUpload(files) {
        if (!this.currentPath) {
            this.showNotification('Выберите папку для загрузки файлов', 'error');
            return;
        }

        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`/api/upload?path=${encodeURIComponent(this.currentPath)}`, {
                method: 'POST',
                body: formData,
                cache: 'no-store'
            });

            const result = await response.json();
            
            let successCount = 0;
            let errorCount = 0;
            
            result.results.forEach(item => {
                if (item.status === 'uploaded' || item.status === 'extracted') {
                    successCount++;
                } else {
                    errorCount++;
                }
            });

            if (successCount > 0) {
                this.showNotification(`Загружено файлов: ${successCount}`, 'success');
                // Обновляем содержимое папки
                this.navigateToFolder(this.currentPath);
            }
            
            if (errorCount > 0) {
                this.showNotification(`Ошибок при загрузке: ${errorCount}`, 'error');
            }

        } catch (error) {
            this.showNotification('Ошибка загрузки файлов: ' + error.message, 'error');
        }

        // Очищаем input
        this.fileInput.value = '';
    }

    async processCommonPhotosAlgorithm() {
        try {
            // Используем initialPath для поиска общих папок
            const rootPath = this.initialPath || this.currentPath;
            if (!rootPath) {
                this.showNotification('Сначала выберите корневую папку для поиска "Общие"', 'error');
                return;
            }

            console.log('🔍 Ищем общие папки в:', rootPath);
            
            // Ищем все общие папки рекурсивно
            const commonFolders = await this.findCommonFoldersRecursive(rootPath);
            
            if (commonFolders.length === 0) {
                this.showNotification('Общие папки не найдены', 'error');
                return;
            }

            console.log('📁 Найдено общих папок:', commonFolders.length);
            
            // Показываем уведомление о начале обработки
            this.showNotification(`Найдено ${commonFolders.length} общих папок. Добавляем в очередь...`, 'success');
            
            // Добавляем все общие папки в очередь с флагом includeExcluded
            let addedCount = 0;
            for (const folderPath of commonFolders) {
                try {
                    console.log('🔍 Добавляем в очередь:', folderPath);
                    const result = await this.addToQueueDirect(folderPath, true);
                    console.log('✅ Добавлена в очередь:', folderPath, result);
                    addedCount++;
                } catch (error) {
                    console.error('❌ Ошибка добавления в очередь:', folderPath, error);
                }
            }
            
            console.log(`📊 Добавлено в очередь: ${addedCount} из ${commonFolders.length} папок`);
            
            if (addedCount === 0) {
                this.showNotification('Не удалось добавить ни одной папки в очередь', 'error');
                return;
            }
            
            // Запускаем обработку очереди
            console.log('🚀 Запускаем обработку очереди...');
            await this.processQueueWithExcluded();
            
        } catch (error) {
            console.error('❌ Ошибка при обработке общих фото:', error);
            this.showNotification('Ошибка при обработке общих фото: ' + error.message, 'error');
        }
    }

    async findCommonFoldersRecursive(rootPath, depth = 0, maxDepth = 3, visitedPaths = new Set()) {
        const commonFolders = [];
        const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
        
        // Проверяем, не посещали ли мы уже этот путь
        if (visitedPaths.has(rootPath)) {
            console.log('⚠️ Пропускаем уже посещенный путь:', rootPath);
            return commonFolders;
        }
        visitedPaths.add(rootPath);
        
        // Ограничиваем глубину поиска
        if (depth > maxDepth) {
            console.log('⚠️ Достигнута максимальная глубина поиска:', depth);
            return commonFolders;
        }
        
        // Список системных папок, которые нужно пропустить
        const systemFolders = [
            '/Applications', '/bin', '/sbin', '/usr', '/var', '/System', '/Library',
            '/private', '/etc', '/tmp', '/opt', '/home', '/root', '/dev', '/proc',
            '/sys', '/mnt', '/media', '/run', '/lost+found', '/Users/Shared'
        ];
        
        // Проверяем, не пытаемся ли мы получить доступ к системной папке
        if (systemFolders.some(sysPath => rootPath.startsWith(sysPath))) {
            console.log('⚠️ Пропускаем системную папку:', rootPath);
            return commonFolders;
        }
        
        // Ограничиваем поиск только в выбранной пользователем папке
        const userSelectedPath = this.initialPath || this.currentPath;
        if (!rootPath.startsWith(userSelectedPath)) {
            console.log('⚠️ Пропускаем папку вне выбранной области:', rootPath);
            return commonFolders;
        }
        
        console.log(`🔍 Поиск на глубине ${depth}: ${rootPath}`);
        
        try {
            const response = await fetch(`/api/folder?path=${encodeURIComponent(rootPath)}&_ts=${Date.now()}`, { cache: 'no-store' });
            
            if (!response.ok) {
                console.log('⚠️ Не удалось получить доступ к папке:', rootPath, 'Статус:', response.status);
                return commonFolders;
            }
            
            const data = await response.json();
            
            if (!data.contents) return commonFolders;
            
            // Проверяем текущий уровень
            for (const item of data.contents) {
                if (item.is_directory) {
                    const folderName = item.name.replace('📂 ', '');
                    const folderNameLower = folderName.toLowerCase();
                    
                    // Пропускаем системные папки
                    if (systemFolders.some(sysPath => item.path.startsWith(sysPath))) {
                        continue;
                    }
                    
                    for (const excludedName of excludedNames) {
                        if (folderNameLower.includes(excludedName)) {
                            commonFolders.push(item.path);
                            console.log('✅ Найдена общая папка:', item.path);
                            break;
                        }
                    }
                }
            }
            
            // Рекурсивно ищем в подпапках (только в безопасных папках)
            for (const item of data.contents) {
                if (item.is_directory) {
                    // Пропускаем системные папки при рекурсивном поиске
                    if (systemFolders.some(sysPath => item.path.startsWith(sysPath))) {
                        continue;
                    }
                    
                    try {
                        const subFolders = await this.findCommonFoldersRecursive(item.path, depth + 1, maxDepth, visitedPaths);
                        commonFolders.push(...subFolders);
                    } catch (error) {
                        console.log('⚠️ Ошибка при рекурсивном поиске в:', item.path, error.message);
                        // Продолжаем поиск в других папках
                    }
                }
            }
            
        } catch (error) {
            console.error('Ошибка при поиске общих папок:', error);
        }
        
        return commonFolders;
    }

    async addExcludedFoldersToQueue() {
        try {
            // Используем initialPath для поиска общих папок
            const rootPath = this.initialPath || this.currentPath;
            if (!rootPath) {
                this.showNotification('Сначала выберите корневую папку для поиска "Общие"', 'error');
                return;
            }

            const response = await fetch(`/api/folder?path=${encodeURIComponent(rootPath)}&_ts=${Date.now()}`, { cache: 'no-store' });
            const data = await response.json();

            const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
            const excludedFolders = [];

            // Данные от бэкенда приходят в поле contents
            const items = Array.isArray(data.contents) ? data.contents : [];

            // Находим все папки с исключаемыми названиями на текущем уровне
            for (const item of items) {
                if (item.is_directory) {
                    const folderName = item.name.replace('📂 ', '');
                    const folderNameLower = folderName.toLowerCase();
                    for (const excludedName of excludedNames) {
                        if (folderNameLower.includes(excludedName)) {
                            excludedFolders.push(item.path);
                            break;
                        }
                    }
                }
            }

            // Добавляем найденные папки в очередь с флагом includeExcluded
            for (const folderPath of excludedFolders) {
                await this.addToQueueDirect(folderPath, true);
            }
            
            if (excludedFolders.length > 0) {
                this.showNotification(`Добавлено ${excludedFolders.length} папок "Общие" в очередь`, 'success');
                await this.loadQueue();
            } else {
                this.showNotification('Папки "Общие" не найдены в текущей директории', 'info');
            }
        } catch (error) {
            this.showNotification('Ошибка поиска папок "Общие": ' + error.message, 'error');
        }
    }

    async addToQueueDirect(path, includeExcluded = false) {
        console.log('🔍 [addToQueueDirect] Начало запроса:', { path, includeExcluded });
        
        try {
            const url = includeExcluded ? '/api/queue/add?includeExcluded=true' : '/api/queue/add';
            console.log('🔍 [addToQueueDirect] URL запроса:', url);
            
            const requestBody = { path: path };
            console.log('🔍 [addToQueueDirect] Тело запроса:', requestBody);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            console.log('🔍 [addToQueueDirect] Статус ответа:', response.status, response.statusText);
            const result = await response.json();
            console.log('🔍 [addToQueueDirect] Ответ сервера:', result);
            
            if (!response.ok) {
                console.error('❌ [addToQueueDirect] Ошибка сервера:', result);
                throw new Error(result.detail || result.message);
            }
            
            console.log('✅ [addToQueueDirect] Успешно добавлено в очередь');
            return result;
        } catch (error) {
            console.error('❌ [addToQueueDirect] Ошибка:', error);
            this.showNotification('Ошибка добавления в очередь: ' + error.message, 'error');
            throw error;
        }
    }

    async addToQueue(path) {
        console.log('🔍 [addToQueue] Начало добавления в очередь:', path);
        
        // Сохраняем корневую директорию для обработки
        this.initialPath = path;
        console.log('🔍 [addToQueue] Сохранен initialPath:', this.initialPath);
        
        // Если не включена обработка исключенных папок, проверяем их
        if (!this.includeExcluded) {
            console.log('🔍 [addToQueue] Проверяем исключенные папки...');
            const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
            const pathLower = path.toLowerCase();
            for (const excludedName of excludedNames) {
                if (pathLower.includes(excludedName)) {
                    console.log('❌ [addToQueue] Папка исключена:', excludedName);
                    this.showNotification(`Папки с названием "${excludedName}" не обрабатываются`, 'error');
                    return;
                }
            }
            console.log('✅ [addToQueue] Папка не исключена, продолжаем...');
        }
        
        try {
            console.log('🔍 [addToQueue] Вызываем addToQueueDirect...');
            const result = await this.addToQueueDirect(path);
            console.log('✅ [addToQueue] Результат addToQueueDirect:', result);
            this.showNotification(result.message, 'success');
            console.log('🔍 [addToQueue] Обновляем очередь...');
            await this.loadQueue();
            console.log('✅ [addToQueue] Очередь обновлена');
        } catch (error) {
            console.error('❌ [addToQueue] Ошибка:', error);
            // Ошибка уже обработана в addToQueueDirect
        }
    }

    async loadQueue() {
        try {
            const response = await fetch('/api/queue', { cache: 'no-store' });
            const data = await response.json();
            this.queue = data.queue;
            this.displayQueue();
        } catch (error) {
            console.error('Ошибка загрузки очереди:', error);
        }
    }

    displayQueue() {
        if (this.queue.length === 0) {
            this.queueList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 20px 0;">
                    Очередь пуста
                </p>
            `;
            this.processBtn.disabled = true;
            this.processGlobalBtn.disabled = true;
            this.clearBtn.disabled = true;
            this.addQueueBtn.disabled = false;
        } else {
            this.queueList.innerHTML = '';
            this.queue.forEach((path, index) => {
                const item = document.createElement('div');
                item.className = 'queue-item';
                item.innerHTML = `
                    <span>${index + 1}. ${path}</span>
                `;
                this.queueList.appendChild(item);
            });
            this.processBtn.disabled = false;
            this.processGlobalBtn.disabled = this.queue.length < 2; // Глобальная обработка требует минимум 2 папки
            this.clearBtn.disabled = false;
            this.addQueueBtn.disabled = false;
        }
    }

    async processQueue() {
        console.log('🔍 [processQueue] Начало обработки очереди');
        
        try {
            console.log('🔍 [processQueue] Отключаем кнопку и показываем загрузку');
            this.processBtn.disabled = true;
            this.processBtn.innerHTML = '<div class="loading"></div> Запуск...';

            // Обновляем очередь перед запуском, чтобы избежать гонки состояний
            console.log('🔍 [processQueue] Обновляем очередь...');
            await this.loadQueue();
            console.log('🔍 [processQueue] Текущая очередь:', this.queue);
            
            if (!this.queue || this.queue.length === 0) {
                console.log('❌ [processQueue] Очередь пуста');
                this.showNotification('Очередь пуста. Добавьте папки перед запуском.', 'error');
                return;
            }

            const url = `/api/process?includeExcluded=${this.includeExcluded}&jointMode=${this.jointMode}&postValidate=${this.postValidate}`;
            console.log(`🔍 [processQueue] Отправляем запрос: ${url}`);
            console.log(`🔍 [processQueue] includeExcluded: ${this.includeExcluded}, jointMode: ${this.jointMode}, postValidate: ${this.postValidate}`);
            
            const response = await fetch(url, { method: 'POST', cache: 'no-store' });
            console.log('🔍 [processQueue] Статус ответа:', response.status, response.statusText);
            
            const result = await response.json();
            console.log('🔍 [processQueue] Ответ сервера:', result);
            
            if (!response.ok) {
                console.error('❌ [processQueue] Ошибка сервера:', result);
                this.showNotification(result.detail || result.message || 'Ошибка при запуске обработки', 'error');
                return;
            }
            
            console.log('✅ [processQueue] Обработка запущена успешно');
            this.showNotification(result.message, 'success');
            
            console.log('🔍 [processQueue] Обновляем очередь после запуска...');
            await this.loadQueue();
            
        } catch (error) {
            console.error('❌ [processQueue] Ошибка:', error);
            this.showNotification('Ошибка запуска обработки: ' + error.message, 'error');
        } finally {
            console.log('🔍 [processQueue] Восстанавливаем кнопку');
            this.processBtn.disabled = false;
            this.processBtn.innerHTML = '🚀 Обработать очередь';
        }
    }

    async processGlobalQueue() {
        console.log('🌍 [processGlobalQueue] Начало глобальной обработки очереди');

        try {
            console.log('🌍 [processGlobalQueue] Отключаем кнопку и показываем загрузку');
            this.processGlobalBtn.disabled = true;
            this.processGlobalBtn.innerHTML = '<div class="loading"></div> Запуск...';

            // Обновляем очередь перед запуском
            console.log('🌍 [processGlobalQueue] Обновляем очередь...');
            await this.loadQueue();
            console.log('🌍 [processGlobalQueue] Текущая очередь:', this.queue);

            if (!this.queue || this.queue.length < 2) {
                console.log('❌ [processGlobalQueue] Недостаточно папок для глобальной кластеризации');
                this.showNotification('Глобальная кластеризация требует минимум 2 папки', 'error');
                return;
            }

            console.log('🌍 [processGlobalQueue] Отправляем запрос на глобальную обработку');
            const response = await fetch('/api/process-global', { method: 'POST', cache: 'no-store' });
            console.log('🌍 [processGlobalQueue] Статус ответа:', response.status, response.statusText);

            const result = await response.json();
            console.log('🌍 [processGlobalQueue] Ответ сервера:', result);

            if (!response.ok) {
                console.error('❌ [processGlobalQueue] Ошибка сервера:', result);
                this.showNotification(result.detail || result.message || 'Ошибка при запуске глобальной обработки', 'error');
                return;
            }

            console.log('✅ [processGlobalQueue] Глобальная обработка запущена успешно');
            this.showNotification(result.message, 'success');

            console.log('🌍 [processGlobalQueue] Обновляем очередь после запуска...');
            await this.loadQueue();

        } catch (error) {
            console.error('❌ [processGlobalQueue] Ошибка:', error);
            this.showNotification('Ошибка запуска глобальной обработки: ' + error.message, 'error');
        } finally {
            console.log('🌍 [processGlobalQueue] Восстанавливаем кнопку');
            this.processGlobalBtn.disabled = false;
            this.processGlobalBtn.innerHTML = '🌍 Глобальная кластеризация';
        }
    }

    async processQueueWithExcluded() {
        console.log('🔍 [processQueueWithExcluded] Начало обработки очереди с включенными исключениями');
        
        try {
            console.log('🔍 [processQueueWithExcluded] Отключаем кнопку и показываем загрузку');
            this.processBtn.disabled = true;
            this.processBtn.innerHTML = '<div class="loading"></div> Запуск...';

            // Обновляем очередь перед запуском
            console.log('🔍 [processQueueWithExcluded] Обновляем очередь...');
            await this.loadQueue();
            console.log('🔍 [processQueueWithExcluded] Текущая очередь:', this.queue);
            
            if (!this.queue || this.queue.length === 0) {
                console.log('❌ [processQueueWithExcluded] Очередь пуста');
                this.showNotification('Очередь пуста. Добавьте папки перед запуском.', 'error');
                return;
            }

            const url = `/api/process?includeExcluded=true&jointMode=${this.jointMode}&postValidate=${this.postValidate}`;
            console.log(`🔍 [processQueueWithExcluded] Отправляем запрос: ${url}`);
            
            const response = await fetch(url, { method: 'POST', cache: 'no-store' });
            console.log('🔍 [processQueueWithExcluded] Статус ответа:', response.status, response.statusText);
            
            const result = await response.json();
            console.log('🔍 [processQueueWithExcluded] Ответ сервера:', result);
            
            if (!response.ok) {
                console.error('❌ [processQueueWithExcluded] Ошибка сервера:', result);
                this.showNotification(result.detail || result.message || 'Ошибка при запуске обработки', 'error');
                return;
            }
            
            console.log('✅ [processQueueWithExcluded] Обработка запущена успешно');
            this.showNotification(result.message, 'success');
            
            console.log('🔍 [processQueueWithExcluded] Обновляем очередь после запуска...');
            await this.loadQueue();
            
        } catch (error) {
            console.error('❌ [processQueueWithExcluded] Ошибка:', error);
            this.showNotification('Ошибка запуска обработки: ' + error.message, 'error');
        } finally {
            console.log('🔍 [processQueueWithExcluded] Восстанавливаем кнопку');
            this.processBtn.disabled = false;
            this.processBtn.innerHTML = '🚀 Обработать очередь';
        }
    }

    async clearQueue() {
        try {
            const response = await fetch('/api/queue', {
                method: 'DELETE',
                cache: 'no-store'
            });

            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue();

        } catch (error) {
            this.showNotification('Ошибка очистки очереди: ' + error.message, 'error');
        }
    }

    async loadTasks() {
        try {
            const response = await fetch('/api/tasks', { cache: 'no-store' });
            const data = await response.json();
            
            // Обновляем только если есть изменения
            const newTasksStr = JSON.stringify(data.tasks);
            if (this.lastTasksStr !== newTasksStr) {
                this.lastTasksStr = newTasksStr;
                this.displayTasks(data.tasks);
            }
            
        } catch (error) {
            console.error('Ошибка загрузки задач:', error);
        }
    }

    displayTasks(tasks) {
        if (!this.tasksList) {
            console.error('tasksList element not found!');
            return;
        }
        
        // Показываем и активные, и завершенные/с ошибкой задачи (последние сверху)
        const allTasks = Array.isArray(tasks) ? tasks.slice() : [];
        if (allTasks.length === 0) {
            this.tasksList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Задач нет
                </p>
            `;
            return;
        }

        this.tasksList.innerHTML = '';
        
        // Порядок: running → pending → error → completed, затем по времени создания (новые сверху)
        const statusOrder = { running: 0, pending: 1, error: 2, completed: 3 };
        allTasks.sort((a, b) => {
            const byStatus = (statusOrder[a.status] ?? 99) - (statusOrder[b.status] ?? 99);
            if (byStatus !== 0) return byStatus;
            return (b.created_at || 0) - (a.created_at || 0);
        });

        // Ограничим отображение до 10 последних задач, чтобы не раздувать список
        const tasksToShow = allTasks.slice(0, 10);

        tasksToShow.forEach(task => {
            const taskEl = document.createElement('div');
            taskEl.className = `task-item ${task.status}`;
            
            const statusEmoji = {
                'pending': '⏳',
                'running': '⚡',
                'completed': '✅',
                'error': '❌'
            };

            let resultHtml = '';
            if (task.status === 'completed' && task.result) {
                resultHtml = `
                    <div class="result-stats">
                        <div class="stat-item">
                            <div class="stat-value moved">${task.result.moved}</div>
                            <div class="stat-label">Перемещено</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value copied">${task.result.copied}</div>
                            <div class="stat-label">Скопировано</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value clusters">${task.result.clusters_count}</div>
                            <div class="stat-label">Кластеров</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value no-faces">${task.result.no_faces_count}</div>
                            <div class="stat-label">Без лиц</div>
                        </div>
                    </div>
                `;
            } else if (task.status === 'error') {
                const errText = task.error || task.message || 'Неизвестная ошибка';
                resultHtml = `
                    <div class="progress-details" style="color:#c0392b;">${errText}</div>
                `;
            }

            let progressHtml = '';
            if (task.status === 'running' || task.status === 'pending') {
                const progress = task.progress || 0;
                progressHtml = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div class="progress-text">${progress}%</div>
                    <div class="progress-details">${task.message || 'Подготовка...'}</div>
                `;
            }

            // Получаем имя задачи из разных возможных полей
            let taskName = task.folder_path || task.path || task.name || 'Неизвестная задача';
            
            // Улучшаем отображение имени папки
            if (taskName && taskName.includes('/')) {
                // Показываем только имя папки, а не полный путь
                const pathParts = taskName.split('/');
                taskName = pathParts[pathParts.length - 1] || taskName;
            }
            
            taskEl.innerHTML = `
                <div class="task-header">
                    <span>${statusEmoji[task.status]} ${taskName}</span>
                    <button class="task-close" data-task-id="${task.id}">×</button>
                </div>
                ${progressHtml}
                ${resultHtml}
            `;
            this.tasksList.appendChild(taskEl);
        });
    }

    async startTaskPolling() {
        setInterval(async () => {
            console.log('🔄 Автообновление задач (5s)...');
            await this.loadTasks();
        }, 5000); // Проверять каждые 5 секунд
    }

    async startFolderPolling() {
        setInterval(async () => {
            // Автоматически обновляем содержимое папки каждые 5 секунд
            if (this.currentPath) {
                console.log('🔄 Автообновление папки (5s):', this.currentPath);
                await this.navigateToFolder(this.currentPath);
            }
        }, 5000); // Обновлять каждые 5 секунд
    }


    async clearCompletedTasks() {
        try {
            const response = await fetch('/api/tasks/clear', {
                method: 'DELETE',
                cache: 'no-store'
            });
            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadTasks();
        } catch (error) {
            this.showNotification('Ошибка очистки завершенных задач: ' + error.message, 'error');
        }
    }

    async moveItem(srcPath, destPath) {
        console.log('🔧 moveItem called:', srcPath, '→', destPath);
        const key = `${srcPath}→${destPath}`;
        if (this.pendingMoves.has(key)) {
            console.log('⏩ Duplicate move ignored for', key);
            return;
        }
        this.pendingMoves.add(key);
        try {
            const response = await fetch(`/api/move?srcPath=${encodeURIComponent(srcPath)}&destPath=${encodeURIComponent(destPath)}`, {
                method: 'POST',
                cache: 'no-store'
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || `Ошибка ${response.status}`);
            }
            
            // Показываем уведомление об успехе
            this.showNotification(result.message || '✅ Файл перемещен', 'success');
            
            // Обновляем UI
            await this.loadQueue();
            await this.refreshCurrentFolder();
        } catch (error) {
            console.error('❌ Move error:', error);
            this.showNotification('Ошибка перемещения: ' + error.message, 'error');
        } finally {
            this.pendingMoves.delete(key);
        }
    }

    async deleteItem(path) {
        if (!confirm('Вы уверены, что хотите удалить этот файл/папку?')) {
            return;
        }
        try {
            const response = await fetch(`/api/delete?path=${encodeURIComponent(path)}`, {
                method: 'DELETE',
                cache: 'no-store'
            });
            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue(); // Обновляем очередь после удаления
        } catch (error) {
            this.showNotification('Ошибка удаления файла/папки: ' + error.message, 'error');
        }
    }

    async showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    async loadQueue() {
        try {
            const response = await fetch('/api/queue', { cache: 'no-store' });
            const data = await response.json();
            this.queue = data.queue;
            this.displayQueue();
        } catch (error) {
            console.error('Ошибка загрузки очереди:', error);
        }
    }

    addContextMenuToElement(element, itemPath, itemName) {
        element.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Не показываем контекстное меню для навигации "Назад"
            if (itemName.includes('⬅️')) return;
            
            this.contextItemPath = itemPath;
            this.contextItemName = itemName;
            
            // Позиционируем меню
            this.contextMenu.style.left = `${e.pageX}px`;
            this.contextMenu.style.top = `${e.pageY}px`;
            this.contextMenu.classList.add('show');
        });
    }

    hideContextMenu() {
        this.contextMenu.classList.remove('show');
    }

    handleContextAction(action) {
        if (!this.contextItemPath) return;
        
        switch(action) {
            case 'rename':
                this.openRenameModal();
                break;
            case 'delete':
                this.deleteItemConfirm();
                break;
        }
    }

    openCreateFolderModal() {
        if (!this.currentPath) {
            this.showNotification('Выберите папку', 'error');
            return;
        }
        this.folderNameInput.value = '';
        this.createFolderModal.classList.add('show');
        setTimeout(() => this.folderNameInput.focus(), 100);
    }

    openRenameModal() {
        // Извлекаем чистое имя без эмодзи
        let cleanName = this.contextItemName
            .replace('📂 ', '')
            .replace('🖼 ', '')
            .replace(' (не обрабатывается)', '');
        
        this.renameInput.value = cleanName;
        this.renameModal.classList.add('show');
        setTimeout(() => {
            this.renameInput.focus();
            this.renameInput.select();
        }, 100);
    }

    closeModal(modalId) {
        document.getElementById(modalId).classList.remove('show');
    }

    async createFolder() {
        const folderName = this.folderNameInput.value.trim();
        
        if (!folderName) {
            this.showNotification('Введите название папки', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/create-folder?path=${encodeURIComponent(this.currentPath)}&name=${encodeURIComponent(folderName)}`, {
                method: 'POST'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Ошибка создания папки');
            }

            this.showNotification(result.message, 'success');
            this.closeModal('createFolderModal');
            await this.refreshCurrentFolder();
        } catch (error) {
            this.showNotification('Ошибка: ' + error.message, 'error');
        }
    }

    async renameItem() {
        const newName = this.renameInput.value.trim();
        
        if (!newName) {
            this.showNotification('Введите новое название', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/rename?oldPath=${encodeURIComponent(this.contextItemPath)}&newName=${encodeURIComponent(newName)}`, {
                method: 'POST'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Ошибка переименования');
            }

            this.showNotification(result.message, 'success');
            this.closeModal('renameModal');
            await this.refreshCurrentFolder();
        } catch (error) {
            this.showNotification('Ошибка: ' + error.message, 'error');
        }
    }

    async deleteItemConfirm() {
        const itemName = this.contextItemName
            .replace('📂 ', '')
            .replace('🖼 ', '');
        
        if (!confirm(`Вы уверены, что хотите удалить "${itemName}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/delete?path=${encodeURIComponent(this.contextItemPath)}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Ошибка удаления');
            }

            this.showNotification(result.message, 'success');
            await this.refreshCurrentFolder();
        } catch (error) {
            this.showNotification('Ошибка: ' + error.message, 'error');
        }
    }

    async refreshCurrentFolder() {
        if (this.currentPath) {
            await this.navigateToFolder(this.currentPath);
        }
    }

    async downloadZip() {
        if (!this.currentPath) {
            this.showNotification('Выберите папку для архивации', 'error');
            return;
        }

        try {
            this.zipBtn.disabled = true;
            this.zipBtn.innerHTML = '<div class="loading"></div> Создание архива...';

            const response = await fetch(`/api/zip?path=${encodeURIComponent(this.currentPath)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Получаем blob из ответа
            const blob = await response.blob();
            
            // Извлекаем имя папки для имени файла
            const folderName = this.currentPath.split(/[/\\]/).pop() || 'archive';
            const filename = `${folderName}.zip`;

            // Создаем ссылку для скачивания
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Очищаем
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.showNotification(`✅ Архив ${filename} успешно создан`, 'success');
        } catch (error) {
            console.error('❌ Zip error:', error);
            this.showNotification('Ошибка создания архива: ' + error.message, 'error');
        } finally {
            this.zipBtn.disabled = false;
            this.zipBtn.innerHTML = '📦 Скачать ZIP';
        }
    }

    // Автообновление
    startTaskPolling() {
        if (this.taskPollingInterval) {
            clearInterval(this.taskPollingInterval);
        }

        if (this.autoRefreshEnabled) {
            this.taskPollingInterval = setInterval(() => {
                this.updateTasks();
            }, 1000); // Обновление каждую секунду
        }
    }

    startFolderPolling() {
        if (this.folderPollingInterval) {
            clearInterval(this.folderPollingInterval);
        }

        if (this.autoRefreshEnabled) {
            this.folderPollingInterval = setInterval(() => {
                if (this.currentPath) {
                    this.loadFolderContents(this.currentPath);
                }
            }, 3000); // Обновление каждые 3 секунды
        }
    }

    async updateTasks() {
        try {
            const response = await fetch('/api/tasks');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const tasks = data.tasks || [];

            // Обновляем список задач
            this.updateTasksDisplay(tasks);

        } catch (error) {
            console.error('Error updating tasks:', error);
            // Не показываем ошибку пользователю, чтобы не спамить
        }
    }

    updateTasksDisplay(tasks) {
        const tasksStr = JSON.stringify(tasks);
        if (tasksStr === this.lastTasksStr) {
            return; // Ничего не изменилось
        }
        this.lastTasksStr = tasksStr;

        if (tasks.length === 0) {
            this.tasksList.innerHTML = '<p style="text-align: center; color: #666; padding: 40px 0;">Задач пока нет</p>';
            this.clearTasksBtn.style.display = 'none';
            return;
        }

        let html = '';
        let hasCompletedTasks = false;

        tasks.forEach(task => {
            const statusClass = task.status.toLowerCase();
            const progressPercent = task.progress || 0;

            html += `
                <div class="task-item ${statusClass}">
                    <div class="task-header">
                        <span class="task-status">${task.status.toUpperCase()}</span>
                        <span>${task.task_id}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPercent}%"></div>
                    </div>
                    <div class="progress-text">${progressPercent}%</div>
                    <div class="progress-details">
                        ${task.folder_path ? `Папка: ${task.folder_path.split(/[/\\\\]/).pop()}` : ''}
                    </div>
                    <div class="task-message">${task.message || ''}</div>
                </div>
            `;

            if (task.status === 'completed' || task.status === 'error') {
                hasCompletedTasks = true;
            }
        });

        this.tasksList.innerHTML = html;
        this.clearTasksBtn.style.display = hasCompletedTasks ? 'inline-block' : 'none';
    }

    async clearCompletedTasks() {
        try {
            this.clearTasksBtn.disabled = true;
            this.clearTasksBtn.innerHTML = '<div class="loading"></div> Очистка...';

            const response = await fetch('/api/tasks/clear', { method: 'POST' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            this.showNotification(`Очищено ${result.message.match(/(\d+)/)?.[1] || 0} завершенных задач`, 'success');

            // Обновляем список задач
            await this.updateTasks();

        } catch (error) {
            console.error('Error clearing tasks:', error);
            this.showNotification('Ошибка очистки задач: ' + error.message, 'error');
        } finally {
            this.clearTasksBtn.disabled = false;
            this.clearTasksBtn.innerHTML = 'Clear completed';
        }
    }

    async manualRefresh() {
        console.log('🔄 Ручное обновление...');

        // Обновляем задачи
        await this.loadTasks();

        // Обновляем текущую папку
        if (this.currentPath) {
            await this.navigateToFolder(this.currentPath);
        }

        this.showNotification('Обновлено', 'success');
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    const app = new PhotoClusterApp();
});