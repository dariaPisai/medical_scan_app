document.addEventListener('DOMContentLoaded', function() {
    // Form toggle functionality for login/register with flip effect
    const showRegisterForm = document.getElementById('show-register-form');
    const showLoginForm = document.getElementById('show-login-form');
    const flipContainer = document.createElement('div');
    const flipper = document.createElement('div');
    const formContent = document.querySelector('.form-content');
    const loginSection = document.getElementById('login-section');
    const registerSection = document.getElementById('register-section');
    
    // Set up the flip container structure if all required elements exist
    if (showRegisterForm && showLoginForm && loginSection && registerSection && formContent) {
        // Wrap sections in flip container
        flipContainer.className = 'form-flip-container';
        flipper.className = 'form-flipper';
        
        // Move the login and register sections inside the flipper
        const parent = loginSection.parentNode;
        parent.insertBefore(flipContainer, loginSection);
        flipContainer.appendChild(flipper);
        
        // Move both sections into the flipper
        flipper.appendChild(loginSection);
        flipper.appendChild(registerSection);
        
        // Show registration form with flip animation
        showRegisterForm.addEventListener('click', function(e) {
            e.preventDefault();
            flipContainer.classList.add('flipped');
            document.title = "Medical Scan App - Register";
            
            // Wait for the flip animation to start before updating height
            setTimeout(updateContainerHeight, 50);
        });
        
        // Show login form with flip animation
        showLoginForm.addEventListener('click', function(e) {
            e.preventDefault();
            flipContainer.classList.remove('flipped');
            document.title = "Medical Scan App - Login";
            
            // Wait for the flip animation to start before updating height
            setTimeout(updateContainerHeight, 50);
        });
    }

    // Add this function to your existing code, after the setup of the flip container
    function updateContainerHeight() {
        // Get current height of active section
        const activeSection = flipContainer.classList.contains('flipped') 
            ? registerSection 
            : loginSection;
        
        // Set the container height to match the active section
        const sectionHeight = activeSection.offsetHeight;
        flipContainer.style.height = sectionHeight + 'px';
    }

    // Call once on page load to set initial height
    setTimeout(updateContainerHeight, 100);

    // Also update on window resize
    window.addEventListener('resize', updateContainerHeight);

    // Password toggle for registration form
    const registerPasswordInput = document.querySelector('.register-password');
    const confirmPasswordInput = document.querySelector('.confirm-password');
    
    function setupPasswordToggle(inputElement) {
        if (!inputElement) return;
        
        const parentGroup = inputElement.closest('.form-group');
        if (!parentGroup) return;
        
        let labelContainer = parentGroup.querySelector('.label-container');
        
        // If there's no label container, create one
        if (!labelContainer) {
            labelContainer = document.createElement('div');
            labelContainer.className = 'label-container';
            
            // Find the label and move it into the container
            const label = parentGroup.querySelector('.form-label');
            if (label) {
                parentGroup.insertBefore(labelContainer, label);
                labelContainer.appendChild(label);
            } else {
                parentGroup.insertBefore(labelContainer, parentGroup.firstChild);
            }
        }
        
        // Create the toggle button
        const toggleBtn = document.createElement('a');
        toggleBtn.href = "#";
        toggleBtn.className = "password-toggle";
        toggleBtn.textContent = "Show";
        labelContainer.appendChild(toggleBtn);
        
        toggleBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (inputElement.type === 'password') {
                inputElement.type = 'text';
                this.textContent = "Hide";
            } else {
                inputElement.type = 'password';
                this.textContent = "Show";
            }
        });
    }
    
    // Setup password toggles for both password fields
    setupPasswordToggle(registerPasswordInput);
    setupPasswordToggle(confirmPasswordInput);
});