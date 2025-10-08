import streamlit as st
from pathlib import Path

def main() -> None:
    # Page settings
    st.set_page_config(
        page_title="TTP Optimisation Tool",
        layout="wide",
        page_icon="🚆",
    )

    # Centered layout
    col1, col2, col3 = st.columns([1, 2, 1])

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMG_FOLDER = PROJECT_ROOT / "images"
    DEFAULT_PL_IMG = IMG_FOLDER / "pn_loco.jpg"

    with col2:
        # Better quality logo from Pacific National site
        st.image(DEFAULT_PL_IMG, use_container_width=True)

        st.markdown(
            """
            <div style='text-align: center; padding-top: 20px;'>
                <h1 style='font-size: 42px; color: #003366;'>Leave Dashboard</h1>
                <h4 style='color: #FFFFFF;'>🚆 Pacific National</h4>
                <p style='margin-top: 30px; font-size: 16px; color: #FFFFFF;'>👈 Use the sidebar to begin navigating through the tool.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()