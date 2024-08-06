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
        initial_summary,
        final_summary,
        comparison_df,
        num_iterations,
        shots,
    ):
        self.initial_summary = initial_summary
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

        # Initial Summary
        self.add_summary_table(story, self.initial_summary, title="Initial Summary of Circuit Layers")

        # Final Summary
        self.add_summary_table(story, self.final_summary, title="Final Summary of Circuit Layers")

        # Comparison Between Initial and Final Summaries
        self.add_comparison_section(story)

        # Add the loss graph
        self.add_loss_graph(story)

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

    def add_summary_table(self, story, summary, title):
        """Add a summary table of the circuit layers and their words."""
        story.append(Paragraph(title, self.styles["Heading2"]))
        story.append(Spacer(1, 20))

        df = pd.DataFrame(summary)
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)

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

    def add_comparison_section(self, story):
        # Add comparison section
        story.append(
            Paragraph("<a name='section3'/>3. Comparison Between Initial and Final Summaries", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        for idx, row in self.comparison_df.iterrows():
            fig, ax = plt.subplots()
            # Generate a probability distribution plot for each word
            ax.bar(row["Initial Zustand"], row["Initial Wahrscheinlichkeit"], color='blue', alpha=0.5, label="Initial")
            ax.bar(row["Final Zustand"], row["Final Wahrscheinlichkeit"], color='red', alpha=0.5, label="Final")
            ax.set_title(f"Probability Distribution for {row['Wort']}")
            ax.set_xlabel("States")
            ax.set_ylabel("Probability")
            ax.legend()

            # Save the plot as an image
            plot_path = os.path.join("var", f"{row['Wort']}_probability_distribution.png")
            plt.savefig(plot_path)
            plt.close()

            # Add the image to the PDF
            story.append(Image(plot_path, width=400, height=300))
            story.append(Spacer(1, 20))

        # Table with comparison data
        table_data = [self.comparison_df.columns.tolist()] + self.comparison_df.values.tolist()
        table = Table(table_data)

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
        """Add a graph of the loss function to the report."""
        plt.figure(figsize=(10, 6))
        plt.title("Loss Function Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        
        # Hier müsste der Verlustverlauf über die Iterationen hinweg geplottet werden
        # Angenommen, dass die Verlustdaten in self.final_summary als 'Loss' enthalten sind
        for summary in self.final_summary:
            if "Loss" in summary:
                plt.plot(range(len(summary["Loss"])), summary["Loss"], label=f"{summary['Wort']} Loss")
        
        plt.legend()
        plt.grid(True)

        # Save the loss graph as an image
        loss_graph_path = os.path.join("var", "loss_graph.png")
        plt.savefig(loss_graph_path)
        plt.close()

        # Add the loss graph to the PDF
        story.append(Image(loss_graph_path, width=500, height=300))
        story.append(Spacer(1, 20))
        story.append(PageBreak())


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
