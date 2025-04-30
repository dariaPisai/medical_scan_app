document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('scan_file');
    const browseButton = document.getElementById('browse-button');
    const filenameDisplay = document.getElementById('filename-display');
    const selectedFileContainer = document.getElementById('selected-file-container');
    const removeFileButton = document.getElementById('remove-file');
    const analyzeButton = document.getElementById('analyze-button');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const form = document.getElementById('upload-form');

    // Check if elements exist
    if (!dropZone || !fileInput || !filenameDisplay || !form) {
        console.error("Essential elements not found!");
        return;
    }

    // Menu toggle functionality
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('expanded');
            document.querySelector('.main-content').classList.toggle('sidebar-expanded');
        });
    }

    // Trigger file input when browse button is clicked
    if (browseButton) {
        browseButton.addEventListener('click', () => {
            fileInput.click();
        });
    }

    // Remove selected file
    if (removeFileButton) {
        removeFileButton.addEventListener('click', () => {
            fileInput.value = '';
            selectedFileContainer.classList.add('hidden');
            analyzeButton.disabled = true;
            filenameDisplay.textContent = '';
            dropZone.classList.remove('file-selected');
        });
    }

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            
            // Validate file type
            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            if (!allowedTypes.includes(file.type)) {
                showAlert('Invalid file type. Please upload PNG, JPG, or JPEG.');
                resetFileInput();
                return;
            }

            // Update UI to show selected file
            filenameDisplay.textContent = file.name;
            selectedFileContainer.classList.remove('hidden');
            dropZone.classList.add('file-selected');
            analyzeButton.disabled = false;

            // Create a preview if needed
            // createImagePreview(file);
        }
    }

    function showAlert(message) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert';
        alertDiv.textContent = message;
        
        // Find flash messages container or create one
        let flashContainer = document.querySelector('.flash-messages');
        if (!flashContainer) {
            flashContainer = document.createElement('div');
            flashContainer.className = 'flash-messages';
            const panelHeader = document.querySelector('.panel-header');
            panelHeader.parentNode.insertBefore(flashContainer, panelHeader.nextSibling);
        }
        
        // Add alert to container
        flashContainer.appendChild(alertDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
            if (flashContainer.children.length === 0) {
                flashContainer.remove();
            }
        }, 5000);
    }

    function resetFileInput() {
        fileInput.value = '';
        selectedFileContainer.classList.add('hidden');
        analyzeButton.disabled = true;
    }

    // Optional: Create image preview
    function createImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const previewContainer = document.createElement('div');
            previewContainer.className = 'file-preview';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'preview-image';
            
            previewContainer.appendChild(img);
            
            // Find where to insert the preview
            const existingPreview = document.querySelector('.file-preview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            selectedFileContainer.appendChild(previewContainer);
        };
        
        reader.readAsDataURL(file);
    }

    // Check if we have analysis results and scroll to them
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer && resultsContainer.querySelector('.results-content')) {
        // Scroll to results if they exist
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});