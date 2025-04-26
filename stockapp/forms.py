from django import forms

class StockForm(forms.Form):
    code=forms.CharField(label='銘柄名',max_length=20)
    start_date=forms.DateField(label='開始日',widget=forms.DateInput(attrs={'type':'date'}))
    end_date=forms.DateField(label='終了日',widget=forms.DateInput(attrs={'type':'date'}))