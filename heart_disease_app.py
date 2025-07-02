import streamlit as st
import pandas as pd
import joblib
import os

st.markdown("""
    <style>
        /* Container geral maior e scroll vertical fixo */
        .main {
            min-height: 720px;
            overflow-y: hidden;
        }
        /* Container da página maior */
        .block-container {
            min-width: 1600px;  /* aumentei de 1400 para 1600 */
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Ajuste nas colunas pra exemplos e resultado grudados nas bordas */
        .css-1lcbmhc.e1fqkh3o3 {  /* classe padrão colunas do Streamlit, pode variar, então ajuste se quebrar */
            gap: 3rem;
            justify-content: space-between;
        }
    </style>
""", unsafe_allow_html=True)

model_path = 'src/saved_models/logistic_regression_model.joblib'
scaler_path = 'src/saved_models/scaler.joblib'
preprocessing_info_path = 'src/saved_models/preprocessing_info.joblib'

model = scaler = preprocessing_info = None

try:
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(preprocessing_info_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        preprocessing_info = joblib.load(preprocessing_info_path)
    else:
        st.error("Arquivos do modelo não encontrados. Execute o notebook primeiro para treinar e salvar o modelo.")
except Exception as e:
    st.error(f"Erro ao carregar os arquivos do modelo: {str(e)}")


def preprocess_user_input(user_data):
    user_df = pd.DataFrame([user_data])

    if 'cp' in user_df.columns:
        cp_dummies = pd.get_dummies(user_df['cp'], prefix='cp')
        user_df = pd.concat([user_df.drop('cp', axis=1), cp_dummies], axis=1)
        for col in ['cp_typical', 'cp_nontypical', 'cp_nonanginal', 'cp_asymptomatic']:
            if col not in user_df.columns:
                user_df[col] = 0

    if 'thal' in user_df.columns:
        thal_dummies = pd.get_dummies(user_df['thal'], prefix='thal')
        user_df = pd.concat([user_df.drop('thal', axis=1), thal_dummies], axis=1)
        for col in ['thal_normal', 'thal_reversable']:
            if col not in user_df.columns:
                user_df[col] = 0

    user_df['age_group'] = pd.cut(user_df['age'], bins=[0, 40, 55, 65, 100],
                                   labels=['jovem', 'meia_idade', 'sênior', 'idoso'])
    age_dummies = pd.get_dummies(user_df['age_group'], prefix='age_group')
    user_df = pd.concat([user_df.drop('age_group', axis=1), age_dummies], axis=1)
    for col in ['age_group_meia_idade', 'age_group_sênior', 'age_group_idoso']:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df['bp_per_age'] = user_df['trestbps'] / user_df['age']
    user_df['hr_per_age'] = user_df['thalach'] / user_df['age']

    if scaler and preprocessing_info:
        num_features = preprocessing_info['numerical_features_to_scale']
        user_df[num_features] = scaler.transform(user_df[num_features])

    if preprocessing_info:
        required_cols = preprocessing_info['feature_names']
        missing = set(required_cols) - set(user_df.columns)
        for col in missing:
            user_df[col] = 0
        user_df = user_df[required_cols]

    return user_df


def predict_heart_disease(user_data):
    if not model:
        st.error("Modelo não carregado.")
        return None
    try:
        input_df = preprocess_user_input(user_data)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]
        return {
            'prediction': 'Doença Cardíaca' if prediction == 1 else 'Sem Doença Cardíaca',
            'probability': probability,
            'risk_level': 'Alto' if probability > 0.7 else 'Médio' if probability > 0.3 else 'Baixo'
        }
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {str(e)}")
        return None


st.title("Previsão de Doença Cardíaca")
st.markdown("Preencha os dados do paciente ou escolha um exemplo para avaliar o risco de doença cardíaca.")

with st.container():
    col_ex, col_form, col_res = st.columns([1, 3, 1])

    with col_ex:
        st.subheader("Exemplos")

        if st.button("Exemplo 1: Jovem saudável"):
            exemplo = {
                'age': 28, 'sex': 1, 'cp': 'asymptomatic', 'trestbps': 118,
                'chol': 180, 'fbs': 0, 'restecg': 0, 'thalach': 185,
                'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 'normal'
            }
            resultado = predict_heart_disease(exemplo)

        if st.button("Exemplo 2: Idoso com risco"):
            exemplo = {
                'age': 70, 'sex': 1, 'cp': 'typical', 'trestbps': 160,
                'chol': 290, 'fbs': 1, 'restecg': 2, 'thalach': 95,
                'exang': 1, 'oldpeak': 3.0, 'slope': 3, 'ca': 2, 'thal': 'reversable'
            }
            resultado = predict_heart_disease(exemplo)

        if st.button("Exemplo 3: Risco Moderado"):
            exemplo = {
                'age': 55, 'sex': 0, 'cp': 'nontypical', 'trestbps': 135,
                'chol': 240, 'fbs': 0, 'restecg': 1, 'thalach': 140,
                'exang': 1, 'oldpeak': 1.5, 'slope': 2, 'ca': 1, 'thal': 'reversable'
            }
            resultado = predict_heart_disease(exemplo)

        if st.button("Exemplo 4: Sem risco"):
            exemplo = {
                'age': 30, 'sex': 0, 'cp': 'asymptomatic', 'trestbps': 110,
                'chol': 170, 'fbs': 0, 'restecg': 0, 'thalach': 180,
                'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 'normal'
            }
            resultado = predict_heart_disease(exemplo)

    with col_form:
        with st.form("form_entrada"):
            st.subheader("Formulário do Paciente")
            col1, col2 = st.columns(2)

            with col1:
                idade = st.number_input("Idade", 20, 100, 50)
                sexo = st.selectbox("Sexo", [(1, "Masculino"), (0, "Feminino")], format_func=lambda x: x[1])[0]
                dor_peito = st.selectbox("Tipo de dor no peito", ["typical", "nontypical", "nonanginal", "asymptomatic"])
                pressao = st.number_input("Pressão arterial em repouso (mmHg)", 80, 220, 120)
                colesterol = st.number_input("Colesterol sérico (mg/dl)", 100, 600, 200)
                glicose = st.selectbox("Glicemia em jejum > 120 mg/dl?", [(0, "Não"), (1, "Sim")], format_func=lambda x: x[1])[0]

            with col2:
                ecg = st.selectbox("ECG em repouso", [(0, "Normal"), (1, "Alteração ST-T"), (2, "Hipertrofia ventricular esquerda")], format_func=lambda x: x[1])[0]
                freq_max = st.number_input("Frequência cardíaca máxima", 50, 220, 150)
                angina = st.selectbox("Angina induzida por exercício?", [(0, "Não"), (1, "Sim")], format_func=lambda x: x[1])[0]
                oldpeak = st.number_input("Depressão ST (oldpeak)", 0.0, 10.0, 0.0, step=0.1)
                inclinacao = st.selectbox("Inclinação do segmento ST", [(1, "Ascendente"), (2, "Plano"), (3, "Descendente")], format_func=lambda x: x[1])[0]
                vasos = st.number_input("Número de vasos principais (0–3)", 0, 3, 0)
                talassemia = st.selectbox("Talassemia", ["normal", "fixed", "reversable"])

            enviar = st.form_submit_button("Prever")

        if enviar:
            dados_usuario = {
                'age': idade, 'sex': sexo, 'cp': dor_peito, 'trestbps': pressao, 'chol': colesterol,
                'fbs': glicose, 'restecg': ecg, 'thalach': freq_max, 'exang': angina,
                'oldpeak': oldpeak, 'slope': inclinacao, 'ca': vasos, 'thal': talassemia
            }
            resultado = predict_heart_disease(dados_usuario)

    with col_res:
        st.subheader("Resultado da Avaliação")
        if 'resultado' in locals() and resultado:
            st.metric("Predição", resultado['prediction'])
            st.metric("Probabilidade", f"{resultado['probability']:.2%}")
            st.metric("Nível de Risco", resultado['risk_level'])
