// Simple copy to clipboard functionality
document.addEventListener('DOMContentLoaded', () => {
    // Copy Snippet functionality
    const copyBtns = document.querySelectorAll('.copy-btn');

    copyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const textToCopy = btn.getAttribute('data-copy');

            navigator.clipboard.writeText(textToCopy).then(() => {
                const icon = btn.querySelector('.material-symbols-rounded');
                icon.textContent = 'check';
                icon.style.color = 'var(--md-sys-color-primary)';

                setTimeout(() => {
                    icon.textContent = 'content_copy';
                    icon.style.color = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });

    // Simple scroll effect for App Bar
    const appBar = document.querySelector('.top-app-bar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            appBar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.5)';
        } else {
            appBar.style.boxShadow = 'none';
        }
    });
});
