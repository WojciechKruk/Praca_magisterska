from PySide6.QtWidgets import (
    QPushButton, QLabel, QHBoxLayout, QFrame, QVBoxLayout, QWidget, QRadioButton, QButtonGroup, QSpinBox
)
from PySide6.QtGui import QFont, QColor, QIcon
from PySide6.QtWidgets import QGraphicsDropShadowEffect
from PySide6.QtCore import Qt

class UIHelpers:

    @staticmethod
    def clear(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                UIHelpers.clear(child.layout())

    @staticmethod
    def create_button(text, command):
        button = QPushButton(text)
        button.setFixedSize(300, 70)
        button.setFont(QFont('Helvetica', 16, QFont.Bold))
        button.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #007AFF;
                border-radius: 35px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(0, 0, 0, 80))
        button.setGraphicsEffect(shadow)

        button.clicked.connect(command)
        return button

    @staticmethod
    def add_back(layout, command):
        back_btn = UIHelpers.create_button("⟵ Powrót", command)
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)

    @staticmethod
    def add_title(layout, text):
        label = QLabel(text)
        label.setFont(UIHelpers.title_font())
        label.setStyleSheet("color: #262626;")
        layout.addWidget(label, alignment=Qt.AlignCenter)

    @staticmethod
    def title_font():
        return QFont('Helvetica', 24, QFont.Bold)

    @staticmethod
    def radio_group(layout, title, options, padding_x=25, frame_extra_width=50):
        # Cała ramka
        group_box = QFrame()
        total_width = 300 + frame_extra_width
        group_box.setFixedWidth(total_width)

        # Tu sztucznie robimy miejsce na napis
        top_padding_for_title = 20

        group_box.setStyleSheet("""
            QFrame {
                border: 2px solid #262626;
                border-radius: 8px;
                background-color: #E5E8EC;
            }
        """)

        # Dodajemy główny layout do środka
        outer_layout = QVBoxLayout(group_box)
        outer_layout.setContentsMargins(20, top_padding_for_title + 10, 20, 20)

        inner_layout = QHBoxLayout()
        group = QButtonGroup()

        for idx, name in enumerate(options):
            radio = QRadioButton(name)
            radio.setFont(QFont('Helvetica', 14))
            radio.setStyleSheet("color: #262626;")
            inner_layout.addWidget(radio)
            group.addButton(radio, idx)
            if idx == 0:
                radio.setChecked(True)

        outer_layout.addLayout(inner_layout)
        outer_layout.addStretch()

        layout.addSpacing(15)
        layout.addWidget(group_box, alignment=Qt.AlignCenter)
        layout.addSpacing(15)

        # Tytuł wkomponowany w ramkę (teraz działa precyzyjnie)
        if title:
            title_label = QLabel(title, group_box)
            title_label.setFont(QFont('Helvetica', 16, QFont.Bold))
            title_label.setStyleSheet(
                "color: #262626; background-color: #E5E8EC; border: 0px solid black; border-radius: 6px; padding: 0px 10px;"
            )
            title_label.adjustSize()

            # Teraz dokładnie pozycjonujemy etykietę w miejscu nakładki
            title_label.move(padding_x, -(title_label.height() // 2) + top_padding_for_title // 2)

        return group, group

    @staticmethod
    def labeled_spinbox(layout, label_text, spinbox):
        frame = QFrame()
        inner = QHBoxLayout()
        label = QLabel(label_text)
        label.setFont(QFont('Helvetica', 14))
        label.setStyleSheet("color: #262626;")
        inner.addWidget(label)
        inner.addWidget(spinbox)
        frame.setLayout(inner)
        layout.addWidget(frame, alignment=Qt.AlignCenter)

    @staticmethod
    def labeled_resolution(layout, spin_x, spin_y):
        frame = QFrame()
        inner = QHBoxLayout()

        lbl = QLabel("Rozdzielczość WxH:")
        lbl.setFont(QFont('Helvetica', 14))
        lbl.setStyleSheet("color: #262626;")
        inner.addWidget(lbl)
        inner.addWidget(spin_x)
        inner.addWidget(QLabel("x"))
        inner.addWidget(spin_y)

        frame.setLayout(inner)
        layout.addWidget(frame, alignment=Qt.AlignCenter)

    @staticmethod
    def add_label(layout, text):
        label = QLabel(text)
        label.setFont(QFont('Helvetica', 16))
        label.setStyleSheet("color: #262626;")
        layout.addWidget(label, alignment=Qt.AlignCenter)

