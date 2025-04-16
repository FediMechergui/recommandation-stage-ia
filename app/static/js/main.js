/**
 * Main JavaScript for Recommandation d'Offres de Stage par IA
 */

// Check if user is logged in (has a profile)
function checkUserProfile() {
    const userId = localStorage.getItem('userId');
    return !!userId;
}

// Update navigation based on user profile
function updateNavigation() {
    const hasProfile = checkUserProfile();
    const recommendationsLink = document.querySelector('a.nav-link[href="/recommendations"]');
    
    if (recommendationsLink) {
        if (!hasProfile) {
            recommendationsLink.classList.add('disabled');
            recommendationsLink.setAttribute('tabindex', '-1');
            recommendationsLink.setAttribute('aria-disabled', 'true');
        } else {
            recommendationsLink.classList.remove('disabled');
            recommendationsLink.removeAttribute('tabindex');
            recommendationsLink.removeAttribute('aria-disabled');
        }
    }
}

// Format date for display
function formatDate(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString('fr-FR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

// Handle API errors
function handleApiError(error, defaultMessage = 'Une erreur est survenue. Veuillez réessayer.') {
    console.error('API Error:', error);
    alert(defaultMessage);
}

// Clear user profile data (logout)
function clearUserProfile() {
    if (confirm('Êtes-vous sûr de vouloir supprimer votre profil ?')) {
        localStorage.removeItem('userId');
        window.location.href = '/';
    }
}

// Initialize tooltips and popovers
function initBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
    updateNavigation();
    initBootstrapComponents();
    
    // Add logout functionality
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', clearUserProfile);
    }
});
