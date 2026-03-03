import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# -------------------------
# CONFIG
# -------------------------

st.set_page_config(page_title="🍎 Analizador Nutricional IA", layout="wide")

st.title("🍽️ Analizador Nutricional con IA")
st.write("Introduce lo que has comido hoy por texto y obtén un análisis nutricional estimado.")

# -------------------------
# INPUT POR TEXTO
# -------------------------

texto_input = st.text_area("✍️ Escribe lo que has comido hoy:")

# -------------------------
# LANGCHAIN + OPENAI
# -------------------------

objetivo = st.radio(
    "🎯 ¿Cuál es tu objetivo?",
    ["Déficit calórico", "Mantenimiento", "Superávit / volumen"],
    horizontal=True
)

with st.form("datos_usuario"):

    st.subheader("👤 Datos personales")

    col1, col2, col3 = st.columns(3)

    with col1:
        sexo = st.selectbox(
            "Sexo",
            ["Hombre", "Mujer"]
        )

    with col2:
        edad = st.number_input(
            "Edad",
            min_value=10,
            max_value=100,
            value=30
        )

    with col3:
        peso = st.number_input(
            "Peso (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=0.5
        )

    submitted = st.form_submit_button("Guardar datos")

if st.button("🔍 Analizar comida") and texto_input:

    with st.spinner("Analizando nutrientes..."):

        # Definir esquema estructurado
        response_schemas = [
            ResponseSchema(
                name="total_kcal",
                description="Número total aproximado de calorías (solo número)"
            ),
            ResponseSchema(
                name="alimentos",
                description="""
                    Lista de objetos con esta estructura exacta:
                    [
                    {
                        "nombre": string,
                        "kcal": number,
                        "proteinas": number,
                        "carbohidratos": number,
                        "grasas": number
                    }
                    ]
                    """
                ),
            ResponseSchema(
                name="resumen",
                description="""
                Breve valoración nutricional del día (máximo 4-5 líneas).
                Debe incluir una sugerencia concreta de mejora si es necesario.
                """
            ),
            ResponseSchema(
                name="puntuacion",
                description="""
                Número entero entre 0 y 100.
                Evalúa la calidad de la dieta según el objetivo y datos del usuario.
                100 = excelente alineación nutricional.
                """
            )
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template("""
        Eres un nutricionista profesional.

        Datos del usuario:
        - Sexo: {sexo}
        - Edad: {edad}
        - Peso: {peso} kg
        - Objetivo: {objetivo}

        1. Calcula valores aproximados.
        2. Devuelve los datos en el formato solicitado.
        3. Valora la dieta teniendo en cuenta los datos personales. Pon notas altas para mejorar estima.
        4. Si no está alineada con el objetivo, sugiere mejoras concretas.

        {format_instructions}

        Comida:
        {comida}
        """)

        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        chain = prompt | model

        response = chain.invoke({
            "comida": texto_input,
            "objetivo": objetivo,
            "sexo": sexo,
            "edad": edad,
            "peso": peso,
            "format_instructions": format_instructions
        })

        parsed = output_parser.parse(response.content)

        # -------------------------
        # RESULTADOS
        # -------------------------

        alimentos = parsed["alimentos"]

        df = pd.DataFrame(alimentos)

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.subheader("🔥 Calorías totales aproximadas")
            st.metric("Kcal totales", parsed["total_kcal"])

            if sexo == "Hombre":
                tmb = 10 * peso + 6.25 * 170 - 5 * edad + 5
            else:
                tmb = 10 * peso + 6.25 * 170 - 5 * edad - 161

            st.metric("🔥 Metabolismo basal estimado", round(tmb), "kcal/día")

            score = int(parsed["puntuacion"])

            st.subheader("🏆 Puntuación del día")

            # Color dinámico según nota
            if score >= 80:
                color = "🟢"
            elif score >= 50:
                color = "🟡"
            else:
                color = "🔴"

            st.markdown(
                f"""
                <div style="text-align:center; font-size:60px; font-weight:bold;">
                    {color} {score}/100
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown("### 🧠 Valoración nutricional")
            st.caption(f"Objetivo seleccionado: {objetivo}")
            st.write(parsed["resumen"])

        with col3:

            st.subheader("🥑 Distribución de macronutrientes")

            total_prot = df["proteinas"].sum()
            total_carb = df["carbohidratos"].sum()
            total_grasas = df["grasas"].sum()

            macros = {
                "Proteínas": total_prot,
                "Carbohidratos": total_carb,
                "Grasas": total_grasas
            }

            fig, ax = plt.subplots()

            # Fondo transparente
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            ax.pie(
                macros.values(),
                labels=macros.keys(),
                autopct="%1.1f%%",
                textprops={"color": "white"}
            )

            st.pyplot(fig, transparent=True)

        st.divider()

        st.subheader("📊 Detalle por alimento")
        st.dataframe(df, use_container_width=True)


                    

