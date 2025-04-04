document.addEventListener("DOMContentLoaded", function () {
    const track = document.querySelector(".slider-track");
    const images = Array.from(track.children);

    // Дублируем изображения для бесконечной карусели
    images.forEach(img => {
        let clone = img.cloneNode(true);
        track.appendChild(clone);
    });

    let scrollSpeed = 1; // Скорость движения (чем больше, тем быстрее)

    function scrollSlider() {
        // Если все изображения прокрутились (достигли конца оригинальных изображений), перемещаем их в начало
        if (track.scrollLeft >= track.scrollWidth / 2) {
            track.scrollLeft = 0; // сбрасываем scrollLeft в 0, чтобы продолжать прокручивать без скачков
        }
        track.scrollLeft += scrollSpeed; // прокручиваем на заданную скорость
        requestAnimationFrame(scrollSlider); // продолжаем анимацию
    }

    scrollSlider();
});
document.addEventListener("DOMContentLoaded", () => {
    const sections = document.querySelectorAll("section");

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
            }
        });
    }, { threshold: 0.1 });

    sections.forEach(section => observer.observe(section));
});
