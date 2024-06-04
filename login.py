import streamlit as st
import streamlit_authenticator as stauth
import streamlit_appV1
# 此行代码被强制要求放在第一行。
st.set_page_config(layout="wide") # 设置屏幕展开方式，宽屏模式布局更好

credentials = {'usernames': {
                'miaoxy': {'email': 'miaoxy2022@gmail.com',
                            'name': 'miaoxy',
                            'password': '123456'},   
                'admin': {'email': 'administrator@gmail.com',
                            'name': '管理员',
                            'password': '123456'} 
                            }
               }

authenticator = stauth.Authenticate(
    credentials,
    'some_cookie_name', 
    'some_signature_key', 
    cookie_expiry_days=30
)

fields = {
    'Form name': 'Login',
    'Username': 'Username',
    'Password': 'Password',
    'Login': 'Login'
}

name, authentication_status, username = authenticator.login('main', fields=fields)

if authentication_status:
    with st.container():
        cols1, cols2 = st.columns([1,12])
        cols1.write('欢迎 *%s*' % (name))
        with cols2.container():
            authenticator.logout('Logout', 'main')

    streamlit_appV1.main()  # 进入业务应用
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')


##streamlit run login.py

