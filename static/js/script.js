// static/js/auth.js (or similar file, linked in base.html)

document.addEventListener('DOMContentLoaded', function() {
    const infoPanel = document.getElementById('login-info-panel');
    const formPanel = document.getElementById('login-form-panel');
    const loginContainer = document.querySelector('.login-container');
    const formContent = formPanel ? formPanel.querySelector('.form-content') : null;
    const formInputs = formPanel ? formPanel.querySelectorAll('.form-input') : [];

    // Store the default gradient
    const defaultGradient = 'linear-gradient(to right, rgba(10, 10, 24, 0.2) 0%, rgba(10, 10, 24, 0.5) 40%, rgba(10, 10, 24, 0.85) 70%, rgba(10, 10, 24, 0.9) 100%)';
    
    // Track the current hover state
    let currentHoverState = 'none'; // 'none', 'info', or 'form'

    function resetToDefault() {
        // Reset all classes and styles
        loginContainer.classList.remove('hover-info-panel', 'hover-form-panel');
        document.documentElement.style.setProperty('--overlay-gradient', defaultGradient);
        currentHoverState = 'none';
    }

    if (infoPanel && formPanel && loginContainer) {
        // Set up initial state
        resetToDefault();
        
        infoPanel.addEventListener('mouseenter', function() {
            resetToDefault(); // First reset to ensure clean state
            
            // When hovering over the info panel (image side)
            loginContainer.classList.add('hover-info-panel');
            document.documentElement.style.setProperty(
                '--overlay-gradient', 
                'linear-gradient(to right, rgba(10, 10, 24, 0.1) 0%, rgba(10, 10, 24, 0.3) 40%, rgba(10, 10, 24, 0.9) 80%)'
            );
            currentHoverState = 'info';
        });
        
        formPanel.addEventListener('mouseenter', function() {
            resetToDefault(); // First reset to ensure clean state
            
            // When hovering over the form panel
            loginContainer.classList.add('hover-form-panel');
            document.documentElement.style.setProperty(
                '--overlay-gradient', 
                'linear-gradient(to right, rgba(10, 10, 24, 0.3) 0%, rgba(10, 10, 24, 0.6) 40%, rgba(10, 10, 24, 0.7) 100%)'
            );
            currentHoverState = 'form';
        });
        
        // Reset when leaving either panel to a non-panel area
        infoPanel.addEventListener('mouseleave', function(e) {
            // Check if we're not entering the form panel
            if (!formPanel.contains(e.relatedTarget)) {
                resetToDefault();
            }
        });
        
        formPanel.addEventListener('mouseleave', function(e) {
            // Check if we're not entering the info panel
            if (!infoPanel.contains(e.relatedTarget)) {
                resetToDefault();
            }
        });
        
        // Reset when mouse leaves the entire container
        loginContainer.addEventListener('mouseleave', function() {
            resetToDefault();
        });
    }

    // Add subtle glow effect to form on focus
    if (formContent && formInputs.length) {
        formInputs.forEach(input => {
            input.addEventListener('focus', function() {
                // Subtle glow on the form when an input is focused
                formContent.style.boxShadow = '0 0 30px rgba(3, 169, 244, 0.2)';
            });

            input.addEventListener('blur', function() {
                // Check if any input is still focused
                const anyFocused = Array.from(formInputs).some(inp => document.activeElement === inp);
                if (!anyFocused) {
                    formContent.style.boxShadow = '';
                }
            });
        });
    }
});