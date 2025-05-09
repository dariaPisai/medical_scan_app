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
        
        // Initial height update
        updateContainerHeight();
    }
    
    // Function to update container height based on current visible content
    function updateContainerHeight() {
        if (!flipContainer) return;
        
        // Determine which section is currently visible
        const currentSection = flipContainer.classList.contains('flipped') ? 
            registerSection : loginSection;
            
        // Reset container height to auto to get proper content height
        flipContainer.style.height = 'auto';
        
        // Set the container height to match the content height
        const contentHeight = currentSection.offsetHeight;
        flipContainer.style.height = contentHeight + 'px';
    }
    
    // Handle password visibility toggles - Fixed version
    const passwordToggles = document.querySelectorAll('.password-toggle');
    
    passwordToggles.forEach(function(toggle) {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Find the password field that belongs to this toggle button
            const formGroup = this.closest('.form-group');
            if (!formGroup) return;
            
            const passwordField = formGroup.querySelector('input[type="password"], input[type="text"]');
            if (!passwordField) return;
            
            // Toggle password visibility
            if (passwordField.type === 'password') {
                passwordField.type = 'text';
                this.textContent = 'Hide';
            } else {
                passwordField.type = 'password';
                this.textContent = 'Show';
            }
        });
    });
    
    // Update container height when window is resized
    window.addEventListener('resize', updateContainerHeight);

    // Reset form fields on page load to clear any browser autofill
    const formInputs = document.querySelectorAll('.form-input');
    formInputs.forEach(input => {
        // Store original placeholder
        const originalPlaceholder = input.placeholder;
        
        // Apply a forced style refresh
        input.style.color = '#202124';
        
        // For password fields, ensure they're of type password
        if (input.id.includes('password')) {
            input.type = 'password';
        }
        
        // Reset placeholder to ensure it's visible
        setTimeout(() => {
            input.placeholder = '';
            setTimeout(() => {
                input.placeholder = originalPlaceholder;
            }, 10);
        }, 10);
    });

    // Add specific CSS to force visibility for Chrome autofill
    const style = document.createElement('style');
    style.textContent = `
        input:-webkit-autofill,
        input:-webkit-autofill:hover, 
        input:-webkit-autofill:focus {
            -webkit-text-fill-color: #202124 !important;
            transition: background-color 5000s ease-in-out 0s !important;
        }
    `;
    document.head.appendChild(style);
});