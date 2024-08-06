import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from qiskit.visualization import plot_histogram
import os
import seaborn as sns


class Visual:
    def __init__(
        self,
        final_summary,
        comparison_df,
        num_iterations,
        shots,
    ):
        self.final_summary = final_summary
        self.comparison_df = comparison_df
        self.num_iterations = num_iterations
        self.shots = shots
        self.styles = getSampleStyleSheet()

    def generate_report(self, filename="QuantumCircuitReport.pdf"):
        # Create the document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Title Page
        self.add_title_page(story)

        # Table of Contents
        self.add_table_of_contents(story)

        # Initiated Data
        self.add_initiated_data(story)

        # Add Loss Graph
        self.add_loss_graph(story)

        # Comparison Between Initial and Final Results
        self.add_comparison_section(story)

        # Build the PDF
        doc.build(story)

    def add_title_page(self, story):
        # Create the title page
        title_page = TitlePage(
            title="LLY-DML",
            subtitle="Part of the LILY Project",
            copyright_info="""Copyright Protection and All Rights Reserved.<br/>
            Contact: <a href="mailto:info@lilyqml.de">info@lilyqml.de</a><br/>
            Website: <a href="http://lilyqml.de">lilyqml.de</a>""",
            description="""<hr/>
            This is LLY-DML, a model of the LILY Quantum Machine Learning Project.<br/>
            Its task is to train datasets to a state using so-called L-Gates, quantum machine learning gates.<br/>
            Input data is used in parts of the machine learning gates, and other phases are optimized so that a state becomes particularly likely.<br/>
            <hr/>""",
            date=datetime.now().strftime("%d.%m.%Y"),
            additional_info="""<b>Date:</b> 01.08.2024<br/>
            <b>Author:</b> LILY Team<br/>
            <b>Version:</b> 1.0<br/>
            <b>Contact:</b> info@lilyqml.de<br/>
            <b>Website:</b> <a href="http://lilyqml.de">lilyqml.de</a><br/>""",
        )
        title_page.build(story, self.styles)

    def add_table_of_contents(self, story):
        toc = TableOfContents(
            contents=[
                "<link href='#section1' color='blue'>1. Initiated Data</link>",
                "<link href='#section2' color='blue'>2. Loss Function</link>",
                "<link href='#section3' color='blue'>3. Comparison Between Initial and Final Results</link>",
            ]
        )
        toc.build(story, self.styles)

    def add_initiated_data(self, story):
        # Add initial data section
        story.append(
            Paragraph("<a name='section1'/>1. Initiated Data", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        # Table with initial data
        data = {
            "Iterations": self.num_iterations,
            "Shots": self.shots,
        }
        data_df = pd.DataFrame([data])
        table = Table([data_df.columns.tolist()] + data_df.values.tolist())
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_loss_graph(self, story):
        # Section 2: Loss Function Graph
        story.append(
            Paragraph("<a name='section2'/>2. Loss Function", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        # Plot Loss Function Graph
        plt.figure(figsize=(10, 6))
        for summary in self.final_summary:
            word = summary["Wort"]
            losses = summary["Loss"]
            plt.plot(losses, label=f"Loss for {word}")

        plt.title("Loss Function Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        loss_graph_path = os.path.join("var", "loss_graph.png")
        plt.savefig(loss_graph_path)
        plt.close()

        # Add loss graph image to the report
        story.append(Image(loss_graph_path, width=500, height=300))
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_comparison_section(self, story):
        # Section 3: Comparison Between Initial and Final Results
        story.append(
            Paragraph(
                "<a name='section3'/>3. Comparison Between Initial and Final Results",
                self.styles["Heading2"],
            )
        )
        story.append(Spacer(1, 20))

        # Convert floats to strings for reportlab compatibility
        comparison_df = self.comparison_df.copy()
        comparison_df["Initial Wahrscheinlichkeit"] = comparison_df[
            "Initial Wahrscheinlichkeit"
        ].apply(lambda x: f"{x:.6f}")
        comparison_df["Final Wahrscheinlichkeit"] = comparison_df[
            "Final Wahrscheinlichkeit"
        ].apply(lambda x: f"{x:.6f}")

        # Add the comparison table
        table_data = [
            [str(i) for i in row] for row in comparison_df.round(4).values.tolist()
        ]  # Convert all elements to strings
        comparison_table = Table([comparison_df.columns.tolist()] + table_data)
        comparison_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(comparison_table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_final_results_section(self, story):
        # Section 4: Final Results
        story.append(
            Paragraph("<a name='section4'/>4. Final Results", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        # Determine the best optimizer
        final_df = pd.DataFrame(self.final_summary)
        final_df["Improvement"] = (
            final_df["Final Wahrscheinlichkeit"] - final_df["Initial Wahrscheinlichkeit"]
        )
        best_optimizer = final_df.loc[
            final_df["Improvement"].idxmax(), "Wort"
        ]

        final_results_content = f"The most effective optimization method was <b>{best_optimizer}</b>, which achieved the highest improvement in target state probability."

        final_results_section = FinalResultsSection(content=final_results_content)
        final_results_section.build(story, self.styles)


class TitlePage:
    def __init__(
        self, title, subtitle, copyright_info, description, date, additional_info
    ):
        self.title = title
        self.subtitle = subtitle
        self.copyright_info = copyright_info
        self.description = description
        self.date = date
        self.additional_info = additional_info

    def build(self, story, styles):
        # Custom styles for specific formatting
        title_style = ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontSize=36,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=24,  # Abstand nach der Hauptüberschrift
        )
        subtitle_style = ParagraphStyle(
            "subtitle",
            parent=styles["Heading2"],
            fontSize=16,
            spaceAfter=5,  # Weniger Abstand zwischen Untertitel und Copyright-Text
        )
        normal_style = ParagraphStyle(
            "normal",
            parent=styles["Normal"],
            alignment=4,  # Blocksatz
            spaceBefore=10,  # Abstand vor dem Text
        )
        big_heading_style = ParagraphStyle(
            "bigHeading",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=20,  # Abstand nach der Überschrift
        )

        # Paragraphs with the new styles
        title_paragraph = Paragraph(self.title, title_style)
        subtitle_paragraph = Paragraph(self.subtitle, subtitle_style)
        copyright_paragraph = Paragraph(self.copyright_info, normal_style)
        description_paragraph = Paragraph(self.description, normal_style)
        big_heading_paragraph = Paragraph(
            "QUANTUM LLY-DML TRAINING REPORT", big_heading_style
        )
        additional_info_paragraph = Paragraph(self.additional_info, normal_style)

        date_paragraph = Paragraph(
            f"""<hr/>
            This report shows all data related to the training conducted on: <b>{self.date}</b>
            <hr/>""",
            normal_style,
        )

        # Adding content to the story
        story.extend(
            [
                title_paragraph,
                subtitle_paragraph,
                copyright_paragraph,
                Spacer(1, 40),  # Adjusted space between copyright and description
                description_paragraph,
                Spacer(1, 20),
                big_heading_paragraph,
                additional_info_paragraph,
                Spacer(1, 40),
                date_paragraph,
                Spacer(1, 40),
                PageBreak(),  # Page break for the table of contents
            ]
        )


class TableOfContents:
    def __init__(self, contents):
        self.contents = contents

    def build(self, story, styles):
        toc_title_style = styles["Heading2"]
        toc_title = Paragraph("Table of Contents", toc_title_style)

        # Create paragraphs for each item in the table of contents
        toc_entries = [toc_title]
        toc_style = ParagraphStyle(
            "toc",
            parent=styles["Normal"],
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            fontSize=12,
        )

        for entry in self.contents:
            toc_entry = Paragraph(entry, toc_style)
            toc_entries.append(toc_entry)

        # Adding content to the story
        story.extend(toc_entries + [Spacer(1, 40)])


class FinalResultsSection:
    def __init__(self, content):
        self.content = content

    def build(self, story, styles):
        final_results_title_style = styles["Heading2"]
        final_results_title = Paragraph("Final Results", final_results_title_style)

        final_results_content = Paragraph(self.content, styles["Normal"])

        # Adding content to the story
        story.extend(
            [
                final_results_title,
                Spacer(1, 20),
                final_results_content,
                Spacer(1, 40),
            ]
        )
