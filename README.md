# ¿Cuánto mejora una previsión de ventas cuando sabe qué estaba pasando *fuera* de la tienda?

En este proyecto he comparado dos modelos de previsión semanal de ventas: SARIMA, basado en datos históricos, y SARIMAX, que incorpora factores externos como promociones, huelgas, conflictos y problemas logísticos. Mi objetivo es analizar cuál ofrece predicciones más precisas y evaluar la fiabilidad de las diferencias entre ambos modelos.

**[➡ Abrir el proyecto](https://forecast-ventas-retail-app-w5msdvcsbuxas94qjcdore.streamlit.app)**

## Contexto

Una cadena retail necesita saber cuánto va a vender cada semana para decidir cuánto stock pedir y cuánto
personal programar. Un modelo que solo mira el pasado no puede saber que esa semana hubo una huelga, así que
comparé dos modelos sobre 156 semanas de ventas y medí no solo cuánto mejora el que conoce el contexto, sino
**de dónde sale esa mejora y cuánto se puede confiar en ella**.


## Qué te encontrarás en este proyecto

1. **El problema**: por qué prever ventas a ciegas sale caro
2. **Los datos**: 156 semanas, 5 señales y cuántos *episodios* hay de cada una
3. **Antes de modelar**: qué se ve a simple vista
4. **El camino hasta el modelo**: un método pensado para no engañarse
5. **Los dos modelos**: el que mira el pasado contra el que mira el contexto
6. **¿De dónde viene la mejora?**: semana a semana, con y sin incidencias
7. **¿Cuánto podemos fiarnos?**: backtesting en 13 ventanas, dónde falla y diagnóstico de residuos
8. **¿Cuánto pesa cada señal?**: qué efectos merecen confianza y cuáles no
9. **Ponlo a prueba**: Playground con avisos cuando el escenario se sale de lo que el histórico ha visto
10. **¿Qué hemos descubierto?**: el resultado, con sus matices
11. **¿Qué podría hacer el negocio?**: hipótesis, no conclusiones
12. **Lo que sabemos y lo que no**: limitaciones y siguientes pasos
13. **Del dato a la decisión**: conclusión

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación y [statsmodels](https://www.statsmodels.org/)
(SARIMAX) para el modelo de series temporales. El análisis completo, en formato notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/04-series-temporales/arima/03-forecast-ventas-retail).


**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
