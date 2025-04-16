document.addEventListener('DOMContentLoaded', function() {
    const passwordInput = document.getElementById('password-input');
    const toggleLink = document.getElementById('password-toggle-link');

    if(passwordInput && toggleLink) {
        toggleLink.addEventListener('click', function (event) {

            event.preventDefault();

            const currentType = passwordInput.getAttribute('type');

            if(currentType == 'password') {
                passwordInput.setAttribute('type', 'text');
                this.textContent = 'Hide';
            } else {
                passwordInput.setAttribute('type', 'password');
                this.textContent = 'Show';
            }
        });
    }

});