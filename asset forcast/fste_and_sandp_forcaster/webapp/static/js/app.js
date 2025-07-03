// Minimal client-side helpers

// Bootstrapped tooltips
window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));

  // Lazy-load images with IntersectionObserver
  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const img = e.target;
      img.src = img.getAttribute('data-src');
      img.onload = () => img.classList.remove('skeleton');
      obs.unobserve(img);
    });
  }, {rootMargin: '200px 0px'});
  document.querySelectorAll('img.lazy-load').forEach(img => io.observe(img));

  // Market filter buttons
  const btnGroup = document.getElementById('marketFilter');
  if (btnGroup) {
    btnGroup.addEventListener('click', evt => {
      if (!evt.target.dataset.filter) return;
      const selected = evt.target.dataset.filter;
      btnGroup.querySelectorAll('button').forEach(b => b.classList.toggle('active', b===evt.target));
      document.querySelectorAll('[data-market]').forEach(card => {
        const show = selected === 'all' || card.dataset.market === selected;
        card.classList.toggle('d-none', !show);
      });
    });
  }
}); 