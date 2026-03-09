// Show loading spinner
function showLoader() {
    document.getElementById('submit-btn').style.display = 'none';
    document.getElementById('loader').style.display = 'flex';
}

// Paste live news into textarea
function pasteNews(text) {
    const textarea = document.getElementById('news-input');
    textarea.value = text;
    
    // Highlight the textarea slightly to show it happened
    textarea.style.borderColor = '#10B981';
    textarea.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.3)';
    
    // Reset after 1s
    setTimeout(() => {
        textarea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        textarea.style.boxShadow = 'none';
    }, 800);
}
