"""Bloques de HTML reutilizables para mantener el look editorial en toda
la app: eyebrows, títulos, KPI tiles, callouts de hallazgo, tarjetas..."""
import streamlit as st
import streamlit.components.v1 as components

PALETTE = {
    "blue": "#2563EB", "green": "#15803D", "red": "#DC2626",
    "amber": "#B45309", "purple": "#6D28D9", "muted": "#7C8896",
}


_NAV_ITEMS = [
    ("contexto", "Contexto"), ("datos", "Datos"), ("exploracion", "Exploración"),
    ("modelo", "Modelo"), ("explicabilidad", "Explicabilidad"), ("playground", "Playground"),
    ("resultados", "Resultados"), ("decisiones", "Decisiones"),
]


def nav():
    links = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in _NAV_ITEMS)
    st.markdown(
        f"""
        <div class="co-nav">
          <span class="co-nav-brand">Previsión de Ventas · Retail</span>
          <div class="co-nav-links">{links}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def install_smooth_scroll():
    """Streamlit renderiza el contenido dentro de un contenedor con su propio
    scroll (`section[data-testid="stMain"]`), así que un `<a href="#id">` normal
    no lo desplaza, y `st.markdown` sanea los atributos `onclick`. Este componente
    inyecta un listener global (vía el iframe del componente, mismo origen) que
    intercepta los clics en enlaces `#ancla` y hace `scrollIntoView` a mano."""
    # Sin guarda de "ya instalado": Streamlit destruye y recrea el iframe de
    # este componente en cada rerun (al mover un slider, por ejemplo), lo que
    # mata el listener anterior. Reinstalar en cada rerun asegura que siempre
    # hay uno activo; los duplicados de reruns previos ya están muertos junto
    # con su iframe, así que no hay efecto doble.
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (doc.__coScrollHandler) {
            doc.removeEventListener('click', doc.__coScrollHandler);
          }
          doc.__coScrollHandler = function(e){
            const a = e.target.closest('a[href^="#"]');
            if (!a) return;
            const id = a.getAttribute('href').slice(1);
            const el = doc.getElementById(id);
            if (el) {
              e.preventDefault();
              el.scrollIntoView({behavior:'instant', block:'start'});
            }
          };
          doc.addEventListener('click', doc.__coScrollHandler);
        })();
        </script>
        """,
        height=0,
    )


def section_open(anchor: str, tight: bool = False):
    cls = "co-section tight" if tight else "co-section"
    st.markdown(f'<div id="{anchor}" class="{cls}">', unsafe_allow_html=True)


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def eyebrow(text: str, muted: bool = False):
    cls = "eyebrow eyebrow-muted" if muted else "eyebrow"
    st.markdown(f'<p class="{cls}">{text}</p>', unsafe_allow_html=True)


def h2(text: str):
    st.markdown(f'<h2 class="co-h2">{text}</h2>', unsafe_allow_html=True)


def h3(text: str):
    st.markdown(f'<h3 class="co-h3">{text}</h3>', unsafe_allow_html=True)


def lead(text: str):
    st.markdown(f'<p class="co-lead">{text}</p>', unsafe_allow_html=True)


def body(text: str):
    st.markdown(f'<p class="co-body">{text}</p>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="co-divider"></div>', unsafe_allow_html=True)


def kpi_grid(items, cols: int = 4):
    """items: list of dicts {num, label, tone(optional: pos/neg)}"""
    cls = "kpi-grid" if cols == 4 else f"kpi-grid cols-{cols}"
    html = [f'<div class="{cls}">']
    for it in items:
        tone = f' {it["tone"]}' if it.get("tone") else ""
        html.append(
            f'<div class="kpi-tile"><div class="kpi-num{tone}">{it["num"]}</div>'
            f'<div class="kpi-label">{it["label"]}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def finding(text: str, tone: str = ""):
    """tone: '', 'green', 'red', 'amber', 'purple', 'neutral'"""
    cls = f"finding {tone}".strip()
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def card_open(paper: bool = False):
    cls = "co-card paper" if paper else "co-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def pipeline(steps):
    html = ['<div class="pipe-row">']
    for i, s in enumerate(steps):
        if i > 0:
            html.append('<span class="pipe-arrow">&#8594;</span>')
        html.append(f'<span class="pipe-step">{s}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def method_cards(steps):
    """steps: list of (num, text)"""
    cols = st.columns(len(steps))
    for c, (num, text) in zip(cols, steps):
        with c:
            st.markdown(
                f'<div class="method-card"><span class="method-num">{num}</span>'
                f'<div class="method-text">{text}</div></div>',
                unsafe_allow_html=True,
            )


def question_block(eyebrow_text: str, question_html: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="question-block">
          <p class="question-eyebrow">{eyebrow_text}</p>
          <p class="question-text">{question_html}</p>
          {f'<p class="question-sub">{sub}</p>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def impact_banner(text_html: str, quote: str = ""):
    st.markdown(
        f"""
        <div class="impact-banner">
          <p class="impact-text">{text_html}</p>
          {f'<p class="impact-quote">{quote}</p>' if quote else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, outline: bool = False):
    cls = "badge outline" if outline else "badge"
    return f'<span class="{cls}">{text}</span>'


def insight_card(num: str, title: str, text: str, so_what: str):
    st.markdown(
        f"""
        <div class="insight-card">
          <div class="insight-num">{num}</div>
          <div class="insight-body">
            <b class="h">{title}</b>
            <p>{text}</p>
            <p class="so-what">&#8594; {so_what}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_flow(insight: str, action: str, objective: str, metric: str):
    st.markdown(
        f"""
        <div class="decision-flow">
          <div class="df-box"><span class="df-k">Insight</span><span class="df-v">{insight}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box"><span class="df-k">Acción</span><span class="df-v">{action}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box"><span class="df-k">Objetivo</span><span class="df-v">{objective}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box"><span class="df-k">Métrica</span><span class="df-v">{metric}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
