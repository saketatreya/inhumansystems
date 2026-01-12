/**
 * AI Generation Artifact Animation
 * Animates SVG filter displacement to create morphing/warping effect
 * that looks like AI video generation artifacts
 */

(function() {
    // Configuration for each layer
    const layers = [
        {
            id: 'ai-layer-1',
            filterId: 'ai-warp-1',
            // Displacement animation parameters
            minScale: 0,
            maxScale: 80,
            cycleTime: 12000, // ms for full cycle
            phaseOffset: 0,
            // Opacity animation
            minOpacity: 0,
            maxOpacity: 0.5
        },
        {
            id: 'ai-layer-2',
            filterId: 'ai-warp-2',
            minScale: 0,
            maxScale: 60,
            cycleTime: 9000,
            phaseOffset: Math.PI * 0.5,
            minOpacity: 0,
            maxOpacity: 0.45
        },
        {
            id: 'ai-layer-3',
            filterId: 'ai-warp-3',
            minScale: 0,
            maxScale: 100, // More aggressive for glitch layer
            cycleTime: 6000,
            phaseOffset: Math.PI,
            minOpacity: 0,
            maxOpacity: 0.4
        },
        {
            id: 'ai-layer-4',
            filterId: 'ai-warp-4',
            minScale: 0,
            maxScale: 50,
            cycleTime: 18000, // Slow deep warp
            phaseOffset: Math.PI * 1.5,
            minOpacity: 0.05,
            maxOpacity: 0.35
        }
    ];

    // Get filter displacement elements
    const filters = {};
    layers.forEach(layer => {
        const filterEl = document.querySelector(`#${layer.filterId} feDisplacementMap`);
        const layerEl = document.getElementById(layer.id);
        if (filterEl && layerEl) {
            filters[layer.id] = {
                displacement: filterEl,
                layer: layerEl,
                config: layer
            };
        }
    });

    // Easing function - makes the emergence feel more organic
    function easeInOutSine(t) {
        return -(Math.cos(Math.PI * t) - 1) / 2;
    }

    // Animation loop
    let startTime = null;
    
    function animate(timestamp) {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;

        // Update each layer
        Object.values(filters).forEach(({ displacement, layer, config }) => {
            // Calculate position in cycle (0 to 1)
            const cycleProgress = ((elapsed / config.cycleTime) + (config.phaseOffset / (Math.PI * 2))) % 1;
            
            // Create emergence-dissolution pattern
            // 0-0.3: emerge from nothing (scale increases, opacity increases)
            // 0.3-0.7: fully materialized (high scale and opacity)
            // 0.7-1.0: dissolve back (scale and opacity decrease)
            
            let intensity;
            if (cycleProgress < 0.15) {
                // Rapid emergence
                intensity = easeInOutSine(cycleProgress / 0.15);
            } else if (cycleProgress < 0.5) {
                // Fully materialized with slight fluctuation
                intensity = 1 - 0.1 * Math.sin((cycleProgress - 0.15) * Math.PI * 5);
            } else if (cycleProgress < 0.65) {
                // Hold
                intensity = 0.9;
            } else {
                // Dissolve
                intensity = easeInOutSine(1 - (cycleProgress - 0.65) / 0.35);
            }

            // Apply displacement scale
            const scale = config.minScale + (config.maxScale - config.minScale) * intensity;
            displacement.setAttribute('scale', scale);

            // Apply opacity
            const opacity = config.minOpacity + (config.maxOpacity - config.minOpacity) * intensity;
            layer.style.opacity = opacity;
        });

        requestAnimationFrame(animate);
    }

    // Add random glitch bursts
    function randomGlitch() {
        const layer3 = filters['ai-layer-3'];
        if (layer3 && Math.random() > 0.7) {
            // Sudden spike in displacement
            layer3.displacement.setAttribute('scale', 150);
            layer3.layer.style.opacity = 0.7;
            
            setTimeout(() => {
                // Quick recovery
                layer3.displacement.setAttribute('scale', 0);
                layer3.layer.style.opacity = 0;
            }, 50 + Math.random() * 100);
        }
        
        // Random interval for next glitch
        setTimeout(randomGlitch, 2000 + Math.random() * 5000);
    }

    // Animate noise baseFrequency for extra organic feel
    function animateNoise() {
        const turbulenceEls = document.querySelectorAll('[id^="turbulence-"]');
        let noiseTime = 0;
        
        function updateNoise() {
            noiseTime += 0.0001;
            turbulenceEls.forEach((el, i) => {
                const baseFreq = parseFloat(el.getAttribute('baseFrequency'));
                // Subtle frequency modulation
                const modulation = 0.002 * Math.sin(noiseTime * (i + 1) * 0.5);
                el.setAttribute('baseFrequency', (baseFreq + modulation).toFixed(4));
            });
            requestAnimationFrame(updateNoise);
        }
        updateNoise();
    }

    // Start animations
    requestAnimationFrame(animate);
    setTimeout(randomGlitch, 3000);
    animateNoise();

    console.log('AI Artifact Animation initialized');
})();
