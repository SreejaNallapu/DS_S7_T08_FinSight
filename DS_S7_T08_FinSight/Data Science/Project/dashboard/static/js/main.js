document.querySelectorAll('.prediction-form').forEach((form) => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (button && form.checkValidity()) {
      button.disabled = true;
      button.textContent = 'Calculating…';
    }
  });
});
