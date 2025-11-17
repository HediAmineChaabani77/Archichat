// ARCHichat - Invoice Chatbot JavaScript

// DOM Elements
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const bgImage = document.getElementById('bgImage');

// Load background image
async function loadBackgroundImage() {
    try {
        const response = await fetch('/api/background-image');
        const data = await response.json();
        if (data.image) {
            bgImage.style.backgroundImage = `url(${data.image})`;
        }
    } catch (error) {
        console.error('Error loading background image:', error);
    }
}

// Initialize
loadBackgroundImage();

// Event Listeners
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) {
        return;
    }
    
    // Disable input
    messageInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.classList.add('processing');
    
    // Show user message
    showUserMessage(message);
    
    // Clear input
    messageInput.value = '';
    
    // Show loading
    showLoading(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur serveur');
        }
        
        // Show bot response
        showBotResponse(data);
        
    } catch (error) {
        showBotError(error.message);
    } finally {
        showLoading(false);
        messageInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.classList.remove('processing');
        messageInput.focus();
    }
});

// Helper Functions

function showUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message fade-in';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar user-avatar';
    avatar.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="8" r="4" fill="white"/>
            <path d="M4 20C4 16.6863 6.68629 14 10 14H14C17.3137 14 20 16.6863 20 20V21H4V20Z" fill="white"/>
        </svg>
    `;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `
        <div class="message-header">
            <strong>Vous</strong>
            <span class="timestamp">${getCurrentTime()}</span>
        </div>
        <p>${escapeHtml(message)}</p>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function convertMarkdownToHTML(text) {
    // Convert markdown **bold** to HTML <strong>bold</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Convert markdown *italic* to HTML <em>italic</em>
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    return text;
}

function showBotResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message fade-in';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="url(#grad1)"/>
            <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <circle cx="9" cy="9" r="1.5" fill="white"/>
            <circle cx="15" cy="9" r="1.5" fill="white"/>
        </svg>
    `;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Convert markdown to HTML and escape HTML to prevent XSS, then convert markdown
    let responseHTML = escapeHtml(data.response);
    responseHTML = convertMarkdownToHTML(responseHTML);
    
    let html = `
        <div class="message-header">
            <strong>🤖 ARCHichat</strong>
            <span class="timestamp">${getCurrentTime()}</span>
        </div>
        <div class="response-text">${responseHTML}</div>
    `;
    
    // Show invoice context and PDF download if available
    if (data.invoice_context && data.pdf_filename) {
        html += `
            <div class="invoice-context">
                <small><strong>📄 Facture actuelle:</strong> ${escapeHtml(data.invoice_context)}</small>
            </div>
            <div class="pdf-download-container" style="margin-top: 12px;">
                <a href="/api/invoice-pdf/${encodeURIComponent(data.pdf_filename)}" 
                   class="pdf-download-btn" 
                   download="${escapeHtml(data.pdf_filename)}"
                   target="_blank">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
                    </svg>
                    <span>Télécharger la facture PDF</span>
                </a>
            </div>
        `;
    }
    
    // Show placement image if requested
    if (data.show_placement_image) {
        // Will be added after innerHTML is set
    }
    
    content.innerHTML = html;
    
    // Load placement image after setting innerHTML
    if (data.show_placement_image) {
        loadPlacementImage(content);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

async function loadPlacementImage(container) {
    try {
        const response = await fetch('/api/placement-image');
        const data = await response.json();
        if (data.image) {
            const imgDiv = document.createElement('div');
            imgDiv.className = 'placement-image-container';
            imgDiv.innerHTML = `
                <p><strong>📍 Localisation des factures :</strong></p>
                <img src="${data.image}" alt="Invoice Placement" class="placement-image">
            `;
            container.appendChild(imgDiv);
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error loading placement image:', error);
    }
}

function showBotError(errorMessage) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message fade-in';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="white"/>
            <path d="M12 8v4M12 16h.01" stroke="#f56565" stroke-width="2" stroke-linecap="round"/>
        </svg>
    `;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `
        <div class="message-header">
            <strong>🤖 ARCHichat</strong>
            <span class="timestamp">${getCurrentTime()}</span>
        </div>
        <p><strong>❌ Erreur</strong></p>
        <div class="error-message">
            ${escapeHtml(errorMessage)}
        </div>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-scroll to bottom on load
window.addEventListener('load', () => {
    scrollToBottom();
});

// Enter key to send (Shift+Enter for new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

